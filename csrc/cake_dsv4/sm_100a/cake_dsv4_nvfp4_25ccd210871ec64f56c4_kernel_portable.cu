/*
 * Copyright (c) 2023 by FlashInfer team.
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


// Portable (public-PTX) QMUL4 for the DSv4 NVFP4 cache: E2M1 x E4M3 -> E4M3 with one rounding.
// Per 16-value block the eight possible magnitudes RN(scale * m), m in {0, 0.5, 1, 1.5, 2, 3, 4, 6}, are built once
// as an 8-byte table (one scale conversion, four exact f16x2 products, four satfinite conversions); each call then
// selects its four bytes with prmt by the E2M1 magnitude code and restores the E2M1 sign with a second prmt.
// Bit-identical to the cvt/mul/cvt form for every finite scale; only a NaN scale differs (sign of the NaN payload).
struct cake_dsv4_qmul4_table_t {
  uint32_t lo;
  uint32_t hi;
};

__device__ __forceinline__ cake_dsv4_qmul4_table_t cake_dsv4_qmul4_table(uint32_t scale) {
  const uint16_t scale_byte = static_cast<uint16_t>(scale & 0xFFu);
  const uint16_t s2 = static_cast<uint16_t>(scale_byte | (scale_byte << 8));
  cake_dsv4_qmul4_table_t t;
  asm("{\n"
      ".reg .b32 sh, c0, c1, c2, c3, p0, p1, p2, p3;\n"
      ".reg .b16 e0, e1, e2, e3;\n"
      "cvt.rn.f16x2.e4m3x2 sh, %2;\n"
      "mov.b32 c0, 0x38000000;\n"  // {0, 0.5}
      "mov.b32 c1, 0x3E003C00;\n"  // {1, 1.5}
      "mov.b32 c2, 0x42004000;\n"  // {2, 3}
      "mov.b32 c3, 0x46004400;\n"  // {4, 6}
      "mul.rn.f16x2 p0, sh, c0;\n"
      "mul.rn.f16x2 p1, sh, c1;\n"
      "mul.rn.f16x2 p2, sh, c2;\n"
      "mul.rn.f16x2 p3, sh, c3;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e0, p0;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e1, p1;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e2, p2;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 e3, p3;\n"
      "mov.b32 %0, {e0, e1};\n"
      "mov.b32 %1, {e2, e3};\n"
      "}\n"
      : "=r"(t.lo), "=r"(t.hi)
      : "h"(s2));
  return t;
}

template <int kVariant>
__device__ __forceinline__ uint32_t cake_dsv4_qmul4_portable(uint32_t src, uint32_t scale) {
  static_assert(kVariant == 5 || kVariant == 6, "invalid Cake DSv4 QMUL4 variant (LOWER4 / HIGHER4 only)");
  const cake_dsv4_qmul4_table_t t = cake_dsv4_qmul4_table(scale);
  const uint32_t h = (kVariant == 5) ? src : (src >> 16);
  const uint32_t sel = h & 0x7777u;  // magnitude codes; prmt reads only the low four selector nibbles
  const uint32_t h4 = h << 4;
  uint32_t mag;
  uint32_t sgn;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(mag) : "r"(t.lo), "r"(t.hi), "r"(sel));
  // replicate mode: output byte k = 8 copies of bit 7 of the selected byte = the sign bit of E2M1 nibble k
  // (nibbles 1 / 3 sit at bits 7 / 15 of h, nibbles 0 / 2 at bits 7 / 15 of h << 4)
  asm("prmt.b32 %0, %1, %2, 0x9D8C;" : "=r"(sgn) : "r"(h), "r"(h4));
  return mag ^ (sgn & 0x80808080u);
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 464
#define TMEM_TMEM_O0_OFFSET 0
#define TMEM_TMEM_O1_OFFSET 128
#define TMEM_TMEM_S_OFFSET 256
#define TMEM_TMEM_SFA0_OFFSET 384
#define TMEM_TMEM_SFA1_OFFSET 400
#define TMEM_TMEM_SFB0_OFFSET 416
#define TMEM_TMEM_SFB1_OFFSET 432
#define TMEM_TMEM_PSUM_OFFSET 448
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QF4_OFF 1024
#define SMEM_SMEM_QF4_STAGE_BYTES 16384
#define SMEM_SMEM_QF4_STRIDE 16384
#define SMEM_SMEM_QSF_OFF 33792
#define SMEM_SMEM_QSF_STAGE_BYTES 2048
#define SMEM_SMEM_QSF_STRIDE 2048
#define SMEM_SMEM_QSF32_OFF 33792
#define SMEM_SMEM_QSF32_STAGE_BYTES 4096
#define SMEM_SMEM_QSF32_STRIDE 4096
#define SMEM_SMEM_QROPE_OFF 37888
#define SMEM_SMEM_QROPE_STAGE_BYTES 16384
#define SMEM_SMEM_QROPE_STRIDE 16384
#define SMEM_SMEM_QSTAGE_OFF 107520
#define SMEM_SMEM_QSTAGE_STAGE_BYTES 16384
#define SMEM_SMEM_QSTAGE_STRIDE 16384
#define SMEM_SMEM_V5_OFF 54272
#define SMEM_SMEM_V5_STAGE_BYTES 16384
#define SMEM_SMEM_V5_STRIDE 16384
#define SMEM_SMEM_V6_OFF 87040
#define SMEM_SMEM_V6_STAGE_BYTES 2048
#define SMEM_SMEM_V6_STRIDE 2048
#define SMEM_SMEM_V7_OFF 87040
#define SMEM_SMEM_V7_STAGE_BYTES 4096
#define SMEM_SMEM_V7_STRIDE 4096
#define SMEM_SMEM_V8_OFF 91136
#define SMEM_SMEM_V8_STAGE_BYTES 16384
#define SMEM_SMEM_V8_STRIDE 16384
#define SMEM_SMEM_V9_OFF 91136
#define SMEM_SMEM_V9_STAGE_BYTES 16384
#define SMEM_SMEM_V9_STRIDE 16384
#define SMEM_SMEM_V10_OFF 173056
#define SMEM_SMEM_V10_STAGE_BYTES 16384
#define SMEM_SMEM_V10_STRIDE 16384
#define SMEM_SMEM_V11_OFF 205824
#define SMEM_SMEM_V11_STAGE_BYTES 2048
#define SMEM_SMEM_V11_STRIDE 2048
#define SMEM_SMEM_V12_OFF 205824
#define SMEM_SMEM_V12_STAGE_BYTES 4096
#define SMEM_SMEM_V12_STRIDE 4096
#define SMEM_SMEM_V13_OFF 209920
#define SMEM_SMEM_V13_STAGE_BYTES 16384
#define SMEM_SMEM_V13_STRIDE 16384
#define SMEM_SMEM_V14_OFF 209920
#define SMEM_SMEM_V14_STAGE_BYTES 16384
#define SMEM_SMEM_V14_STRIDE 16384
#define SMEM_SMEM_KZONE32_OFF 54272
#define SMEM_SMEM_KZONE32_STAGE_BYTES 172032
#define SMEM_SMEM_KZONE32_STRIDE 172032
#define SMEM_SMEM_V_OFF 107520
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_ONES_OFF 226304
#define SMEM_SMEM_ONES_STAGE_BYTES 2048
#define SMEM_SMEM_ONES_STRIDE 2048
#define SMEM_SMEM_ONES32_OFF 226304
#define SMEM_SMEM_ONES32_STAGE_BYTES 2048
#define SMEM_SMEM_ONES32_STRIDE 2048
#define SMEM_SMEM_MASK_OFF 228864
#define SMEM_SMEM_MASK_STAGE_BYTES 64
#define SMEM_SMEM_MASK_STRIDE 64
#define SMEM_SMEM_TOK_OFF 228352
#define SMEM_SMEM_TOK_STAGE_BYTES 512
#define SMEM_SMEM_TOK_STRIDE 512
#define SMEM_SMEM_PMAX_OFF 228928
#define SMEM_SMEM_PMAX_STAGE_BYTES 1536
#define SMEM_SMEM_PMAX_STRIDE 1536
#define SMEM_SMEM_PSUM_OFF 230464
#define SMEM_SMEM_PSUM_STAGE_BYTES 1536
#define SMEM_SMEM_PSUM_STRIDE 1536
#define SMEM_TOTAL 232064
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

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}




__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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




__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo512(int lo) {
    const int SBO = 512;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}



__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


extern "C" {

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_cake_dsv4_nvfp4_25ccd210871ec64f56c4(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_out, __nv_bfloat16* __restrict__ q_rows, uint8_t* __restrict__ main_cache, uint8_t* __restrict__ extra_cache, int* __restrict__ main_indices, int* __restrict__ extra_indices, int* __restrict__ main_lengths, int* __restrict__ extra_lengths, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ lse_out, int num_heads, int num_head_tiles, int num_splits, int num_main_tiles, int tiles_per_split, int total_tiles, int main_width, int extra_width, int main_index_stride, int extra_index_stride, int has_main_lengths, int has_extra_lengths, int main_page_shift, int extra_page_shift, long long main_page_stride, long long extra_page_stride, int has_sinks, float lse_partial_scale, float lse_scale)
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
    #define q_rope_full_addr (mbar_base + 0)
    #define q_nope_full0_addr (mbar_base + 8)
    #define q_nope_full1_addr (mbar_base + 16)
    #define q_nope_full2_addr (mbar_base + 24)
    #define q_ready_addr (mbar_base + 32)
    #define q_ready_k_addr (mbar_base + 40)
    #define gather_landed_k_addr (mbar_base + 48)
    #define gather_landed_p_addr (mbar_base + 64)
    #define tile_meta_addr (mbar_base + 80)
    #define kv_landed_addr (mbar_base + 112)
    #define peer_free_k_addr (mbar_base + 128)
    #define peer_free_p_addr (mbar_base + 144)
    #define kbuf_free_k_addr (mbar_base + 160)
    #define kbuf_free_p_addr (mbar_base + 176)
    #define kv_full_addr (mbar_base + 192)
    #define s_full_addr (mbar_base + 208)
    #define s_free_addr (mbar_base + 224)
    #define k_ready_addr (mbar_base + 232)
    #define p_full_addr (mbar_base + 248)
    #define psum_full_addr (mbar_base + 264)
    #define pv_done_addr (mbar_base + 280)
    #define tmem_dealloc_addr (mbar_base + 296)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_qf4 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_qf4_addr = smem + 1024;
    uint8_t* smem_qsf = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_qsf_addr = smem + 33792;
    unsigned int* smem_qsf32 = reinterpret_cast<unsigned int*>(smem_raw + 33792);
    const int smem_qsf32_addr = smem + 33792;
    __nv_bfloat16* smem_qrope = reinterpret_cast<__nv_bfloat16*>(smem_raw + 37888);
    const int smem_qrope_addr = smem + 37888;
    __nv_bfloat16* smem_qstage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 107520);
    const int smem_qstage_addr = smem + 107520;
    uint8_t* smem_v5 = reinterpret_cast<uint8_t*>(smem_raw + 54272);
    const int smem_v5_addr = smem + 54272;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 87040);
    const int smem_v6_addr = smem + 87040;
    unsigned int* smem_v7 = reinterpret_cast<unsigned int*>(smem_raw + 87040);
    const int smem_v7_addr = smem + 87040;
    __nv_bfloat16* smem_v8 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
    const int smem_v8_addr = smem + 91136;
    uint8_t* smem_v9 = reinterpret_cast<uint8_t*>(smem_raw + 91136);
    const int smem_v9_addr = smem + 91136;
    uint8_t* smem_v10 = reinterpret_cast<uint8_t*>(smem_raw + 173056);
    const int smem_v10_addr = smem + 173056;
    uint8_t* smem_v11 = reinterpret_cast<uint8_t*>(smem_raw + 205824);
    const int smem_v11_addr = smem + 205824;
    unsigned int* smem_v12 = reinterpret_cast<unsigned int*>(smem_raw + 205824);
    const int smem_v12_addr = smem + 205824;
    __nv_bfloat16* smem_v13 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 209920);
    const int smem_v13_addr = smem + 209920;
    uint8_t* smem_v14 = reinterpret_cast<uint8_t*>(smem_raw + 209920);
    const int smem_v14_addr = smem + 209920;
    unsigned int* smem_kzone32 = reinterpret_cast<unsigned int*>(smem_raw + 54272);
    const int smem_kzone32_addr = smem + 54272;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 107520);
    const int smem_v_addr = smem + 107520;
    uint8_t* smem_ones = reinterpret_cast<uint8_t*>(smem_raw + 226304);
    const int smem_ones_addr = smem + 226304;
    unsigned int* smem_ones32 = reinterpret_cast<unsigned int*>(smem_raw + 226304);
    const int smem_ones32_addr = smem + 226304;
    unsigned int* smem_mask = reinterpret_cast<unsigned int*>(smem_raw + 228864);
    const int smem_mask_addr = smem + 228864;
    int* smem_tok = reinterpret_cast<int*>(smem_raw + 228352);
    const int smem_tok_addr = smem + 228352;
    float* smem_pmax = reinterpret_cast<float*>(smem_raw + 228928);
    const int smem_pmax_addr = smem + 228928;
    float* smem_psum = reinterpret_cast<float*>(smem_raw + 230464);
    const int smem_psum_addr = smem + 230464;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_out))) : "memory");

    // Mbarrier init (22 pipeline groups, 0 ordered-sequence groups, 38 barriers)
    // Mbarriers at smem_raw[0..304)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_rope_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_nope_full0: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // q_nope_full1: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // q_nope_full2: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // q_ready: 1 barriers, init_count=384
            mbarrier_init(smem + 32, 384);
            // q_ready_k: 1 barriers, init_count=384
            mbarrier_init(smem + 40, 384);
            // gather_landed_k: 2 barriers, init_count=96
            mbarrier_init(smem + 48, 96);
            mbarrier_init(smem + 56, 96);
            // gather_landed_p: 2 barriers, init_count=96
            mbarrier_init(smem + 64, 96);
            mbarrier_init(smem + 72, 96);
            // tile_meta: 4 barriers, init_count=64
            mbarrier_init(smem + 80, 64);
            mbarrier_init(smem + 88, 64);
            mbarrier_init(smem + 96, 64);
            mbarrier_init(smem + 104, 64);
            // kv_landed: 2 barriers, init_count=2
            mbarrier_init(smem + 112, 2);
            mbarrier_init(smem + 120, 2);
            // peer_free_k: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // peer_free_p: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // kbuf_free_k: 2 barriers, init_count=384
            mbarrier_init(smem + 160, 384);
            mbarrier_init(smem + 168, 384);
            // kbuf_free_p: 2 barriers, init_count=384
            mbarrier_init(smem + 176, 384);
            mbarrier_init(smem + 184, 384);
            // kv_full: 2 barriers, init_count=384
            mbarrier_init(smem + 192, 384);
            mbarrier_init(smem + 200, 384);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // s_free: 1 barriers, init_count=384
            mbarrier_init(smem + 224, 384);
            // k_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 232, 128);
            mbarrier_init(smem + 240, 128);
            // p_full: 2 barriers, init_count=384
            mbarrier_init(smem + 248, 384);
            mbarrier_init(smem + 256, 384);
            // psum_full: 2 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // pv_done: 2 barriers, init_count=1
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            // tmem_dealloc: 1 barriers, init_count=384
            mbarrier_init(smem + 296, 384);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 464 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 304);
    if (warp == 0) {
        int _tmem_hold = smem + 304;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_o0 = taddr;
    const int tmem_tmem_o1 = taddr + 128;
    const int tmem_tmem_s = taddr + 256;
    const int tmem_tmem_sfa0 = taddr + 384;
    const int tmem_tmem_sfa1 = taddr + 400;
    const int tmem_tmem_sfb0 = taddr + 416;
    const int tmem_tmem_sfb1 = taddr + 432;
    const int tmem_tmem_psum = taddr + 448;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    }

    // ---- Role: compute0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // compute0_main
            const int local_warp = warp;
            int rank = cta_rank;
            int work_idx = blockIdx.x / 2;
            int head_tile = work_idx % num_head_tiles;
            int split_work = work_idx / num_head_tiles;
            int split_idx = split_work % num_splits;
            int query_idx = split_work / num_splits;
            int head_base = head_tile * 128;
            const int row = local_warp * 32 + lane;
            int head_row = head_base + row;
            int row_valid = ((head_row < num_heads) ? 1 : 0);
            int warp_rows_valid = ((head_base + local_warp * 32 < num_heads) ? 1 : 0);
            const int tmem_row_origin = local_warp * 32;
            int tile_lo = split_idx * tiles_per_split;
            int tile_hi = tile_lo + tiles_per_split;
            if (tile_hi > total_tiles) {
                tile_hi = total_tiles;
            }
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            float sink_log2 = 0.0f;
            int has_sink_row = 0;
            if (has_sinks != 0 && split_idx == 0 && row_valid != 0) {
                has_sink_row = 1;
                sink_log2 = sinks[head_row] * 1.4426950408889634f;
            }
            {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_ones32_addr + (unsigned int)(row * 16)), "r"(943208504), "r"(943208504), "r"(943208504), "r"(943208504) : "memory");
            }
            float inv_six = 0.16666666666666666f;
            unsigned int _phase_q_nope_full0_0 = 0;
            mbarrier_wait_hint(q_nope_full0_addr, _phase_q_nope_full0_0, 10000000);
            _phase_q_nope_full0_0 ^= 1;
            unsigned int _phase_q_nope_full1_0 = 0;
            mbarrier_wait_hint(q_nope_full1_addr, _phase_q_nope_full1_0, 10000000);
            _phase_q_nope_full1_0 ^= 1;
            unsigned int _phase_q_nope_full2_0 = 0;
            mbarrier_wait_hint(q_nope_full2_addr, _phase_q_nope_full2_0, 10000000);
            _phase_q_nope_full2_0 ^= 1;
            const int q_warp = warp;
            int unit = 0;
            int q_block = 0;
            int kset = 0;
            int q_row = 0;
            int q_row_addr = 0;
            for (int i = 0; i < 3; i++) {
                unit = q_warp + 12 * i;
                if (unit < 28) {
                    q_block = ((unit < 12) ? unit % 4 : (unit - 12) % 4);
                    kset = ((unit < 12) ? 4 + unit / 4 : (unit - 12) / 4);
                    q_row = q_block * 32 + lane;
                    if (head_base + q_block * 32 < num_heads) {
                        q_row_addr = smem_qstage_addr + (unsigned int)(kset * 16384) + (unsigned int)(q_row * 128);
                        unsigned int words[8];
                        unsigned int sf_word = 0;
                        unsigned int qa[4];
                        unsigned int qb[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 3]))
                            : "r"(q_row_addr + (0 ^ q_row % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 3]))
                            : "r"(q_row_addr + (1 ^ q_row % 8) * 16));
                        float qv[16];
                        qv[0] = __uint_as_float(qa[0] << 16);
                        qv[1] = __uint_as_float(qa[0] & 4294901760u);
                        qv[8] = __uint_as_float(qb[0] << 16);
                        qv[9] = __uint_as_float(qb[0] & 4294901760u);
                        qv[2] = __uint_as_float(qa[1] << 16);
                        qv[3] = __uint_as_float(qa[1] & 4294901760u);
                        qv[10] = __uint_as_float(qb[1] << 16);
                        qv[11] = __uint_as_float(qb[1] & 4294901760u);
                        qv[4] = __uint_as_float(qa[2] << 16);
                        qv[5] = __uint_as_float(qa[2] & 4294901760u);
                        qv[12] = __uint_as_float(qb[2] << 16);
                        qv[13] = __uint_as_float(qb[2] & 4294901760u);
                        qv[6] = __uint_as_float(qa[3] << 16);
                        qv[7] = __uint_as_float(qa[3] & 4294901760u);
                        qv[14] = __uint_as_float(qb[3] << 16);
                        qv[15] = __uint_as_float(qb[3] & 4294901760u);
                        float m8[8];
                        float _fabs_0 = fabsf(qv[0]);
                        float _fabs_1 = fabsf(qv[1]);
                        float _max_0 = max_noftz(_fabs_0, _fabs_1);
                        m8[0] = _max_0;
                        float _fabs_2 = fabsf(qv[2]);
                        float _fabs_3 = fabsf(qv[3]);
                        float _max_1 = max_noftz(_fabs_2, _fabs_3);
                        m8[1] = _max_1;
                        float _fabs_4 = fabsf(qv[4]);
                        float _fabs_5 = fabsf(qv[5]);
                        float _max_2 = max_noftz(_fabs_4, _fabs_5);
                        m8[2] = _max_2;
                        float _fabs_6 = fabsf(qv[6]);
                        float _fabs_7 = fabsf(qv[7]);
                        float _max_3 = max_noftz(_fabs_6, _fabs_7);
                        m8[3] = _max_3;
                        float _fabs_8 = fabsf(qv[8]);
                        float _fabs_9 = fabsf(qv[9]);
                        float _max_4 = max_noftz(_fabs_8, _fabs_9);
                        m8[4] = _max_4;
                        float _fabs_10 = fabsf(qv[10]);
                        float _fabs_11 = fabsf(qv[11]);
                        float _max_5 = max_noftz(_fabs_10, _fabs_11);
                        m8[5] = _max_5;
                        float _fabs_12 = fabsf(qv[12]);
                        float _fabs_13 = fabsf(qv[13]);
                        float _max_6 = max_noftz(_fabs_12, _fabs_13);
                        m8[6] = _max_6;
                        float _fabs_14 = fabsf(qv[14]);
                        float _fabs_15 = fabsf(qv[15]);
                        float _max_7 = max_noftz(_fabs_14, _fabs_15);
                        m8[7] = _max_7;
                        float m4[4];
                        float _max_8 = max_noftz(m8[0], m8[1]);
                        m4[0] = _max_8;
                        float _max_9 = max_noftz(m8[2], m8[3]);
                        m4[1] = _max_9;
                        float _max_10 = max_noftz(m8[4], m8[5]);
                        m4[2] = _max_10;
                        float _max_11 = max_noftz(m8[6], m8[7]);
                        m4[3] = _max_11;
                        float _max_12 = max_noftz(m4[0], m4[1]);
                        float _max_13 = max_noftz(m4[2], m4[3]);
                        float _max_14 = max_noftz(_max_12, _max_13);
                        float amax = _max_14;
                        float sc = amax * inv_six;
                        uint16_t _e4m3x2_f32_0;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(0.0f), "f"(sc));
                        uint16_t sc_pair = _e4m3x2_f32_0;
                        unsigned int sc_byte = (unsigned int)sc_pair & 255;
                        unsigned int sc_exp = sc_byte >> 3 & 15;
                        unsigned int sc_man = sc_byte & 7;
                        float sc_norm = __uint_as_float(sc_exp + 120 << 23 | sc_man << 20);
                        float sc_sub = (float)sc_man * 0.001953125f;
                        float sc_dec = ((sc_exp == 0) ? sc_sub : sc_norm);
                        float _rcp_0 = __frcp_rn(sc_dec);
                        float inv = ((sc_dec > 0.0f) ? _rcp_0 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_0 = {inv, inv};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv)[_ls], _scale2_0);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv[_ls] = qv[_ls] * inv;
                        }
                        #endif
                        uint32_t _fp4_pair_0;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_0) : "f"(qv[0]), "f"(qv[1]));
                        uint32_t _fp4_pair_1;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_1) : "f"(qv[2]), "f"(qv[3]));
                        uint32_t _fp4_pair_2;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_2) : "f"(qv[4]), "f"(qv[5]));
                        uint32_t _fp4_pair_3;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_3) : "f"(qv[6]), "f"(qv[7]));
                        uint32_t _fp4_pair_4;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_4) : "f"(qv[8]), "f"(qv[9]));
                        uint32_t _fp4_pair_5;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_5) : "f"(qv[10]), "f"(qv[11]));
                        uint32_t _fp4_pair_6;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_6) : "f"(qv[12]), "f"(qv[13]));
                        uint32_t _fp4_pair_7;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_7) : "f"(qv[14]), "f"(qv[15]));
                        words[0] = _fp4_pair_0 | _fp4_pair_1 << 8 | _fp4_pair_2 << 16 | _fp4_pair_3 << 24;
                        words[1] = _fp4_pair_4 | _fp4_pair_5 << 8 | _fp4_pair_6 << 16 | _fp4_pair_7 << 24;
                        sf_word = sf_word | sc_byte;
                        unsigned int qa_0[4];
                        unsigned int qb_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 3]))
                            : "r"(q_row_addr + (2 ^ q_row % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 3]))
                            : "r"(q_row_addr + (3 ^ q_row % 8) * 16));
                        float qv_2[16];
                        qv_2[0] = __uint_as_float(qa_0[0] << 16);
                        qv_2[1] = __uint_as_float(qa_0[0] & 4294901760u);
                        qv_2[8] = __uint_as_float(qb_1[0] << 16);
                        qv_2[9] = __uint_as_float(qb_1[0] & 4294901760u);
                        qv_2[2] = __uint_as_float(qa_0[1] << 16);
                        qv_2[3] = __uint_as_float(qa_0[1] & 4294901760u);
                        qv_2[10] = __uint_as_float(qb_1[1] << 16);
                        qv_2[11] = __uint_as_float(qb_1[1] & 4294901760u);
                        qv_2[4] = __uint_as_float(qa_0[2] << 16);
                        qv_2[5] = __uint_as_float(qa_0[2] & 4294901760u);
                        qv_2[12] = __uint_as_float(qb_1[2] << 16);
                        qv_2[13] = __uint_as_float(qb_1[2] & 4294901760u);
                        qv_2[6] = __uint_as_float(qa_0[3] << 16);
                        qv_2[7] = __uint_as_float(qa_0[3] & 4294901760u);
                        qv_2[14] = __uint_as_float(qb_1[3] << 16);
                        qv_2[15] = __uint_as_float(qb_1[3] & 4294901760u);
                        float m8_3[8];
                        float _fabs_16 = fabsf(qv_2[0]);
                        float _fabs_17 = fabsf(qv_2[1]);
                        float _max_15 = max_noftz(_fabs_16, _fabs_17);
                        m8_3[0] = _max_15;
                        float _fabs_18 = fabsf(qv_2[2]);
                        float _fabs_19 = fabsf(qv_2[3]);
                        float _max_16 = max_noftz(_fabs_18, _fabs_19);
                        m8_3[1] = _max_16;
                        float _fabs_20 = fabsf(qv_2[4]);
                        float _fabs_21 = fabsf(qv_2[5]);
                        float _max_17 = max_noftz(_fabs_20, _fabs_21);
                        m8_3[2] = _max_17;
                        float _fabs_22 = fabsf(qv_2[6]);
                        float _fabs_23 = fabsf(qv_2[7]);
                        float _max_18 = max_noftz(_fabs_22, _fabs_23);
                        m8_3[3] = _max_18;
                        float _fabs_24 = fabsf(qv_2[8]);
                        float _fabs_25 = fabsf(qv_2[9]);
                        float _max_19 = max_noftz(_fabs_24, _fabs_25);
                        m8_3[4] = _max_19;
                        float _fabs_26 = fabsf(qv_2[10]);
                        float _fabs_27 = fabsf(qv_2[11]);
                        float _max_20 = max_noftz(_fabs_26, _fabs_27);
                        m8_3[5] = _max_20;
                        float _fabs_28 = fabsf(qv_2[12]);
                        float _fabs_29 = fabsf(qv_2[13]);
                        float _max_21 = max_noftz(_fabs_28, _fabs_29);
                        m8_3[6] = _max_21;
                        float _fabs_30 = fabsf(qv_2[14]);
                        float _fabs_31 = fabsf(qv_2[15]);
                        float _max_22 = max_noftz(_fabs_30, _fabs_31);
                        m8_3[7] = _max_22;
                        float m4_4[4];
                        float _max_23 = max_noftz(m8_3[0], m8_3[1]);
                        m4_4[0] = _max_23;
                        float _max_24 = max_noftz(m8_3[2], m8_3[3]);
                        m4_4[1] = _max_24;
                        float _max_25 = max_noftz(m8_3[4], m8_3[5]);
                        m4_4[2] = _max_25;
                        float _max_26 = max_noftz(m8_3[6], m8_3[7]);
                        m4_4[3] = _max_26;
                        float _max_27 = max_noftz(m4_4[0], m4_4[1]);
                        float _max_28 = max_noftz(m4_4[2], m4_4[3]);
                        float _max_29 = max_noftz(_max_27, _max_28);
                        float amax_5 = _max_29;
                        float sc_6 = amax_5 * inv_six;
                        uint16_t _e4m3x2_f32_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(0.0f), "f"(sc_6));
                        uint16_t sc_pair_7 = _e4m3x2_f32_1;
                        unsigned int sc_byte_8 = (unsigned int)sc_pair_7 & 255;
                        unsigned int sc_exp_9 = sc_byte_8 >> 3 & 15;
                        unsigned int sc_man_10 = sc_byte_8 & 7;
                        float sc_norm_11 = __uint_as_float(sc_exp_9 + 120 << 23 | sc_man_10 << 20);
                        float sc_sub_12 = (float)sc_man_10 * 0.001953125f;
                        float sc_dec_13 = ((sc_exp_9 == 0) ? sc_sub_12 : sc_norm_11);
                        float _rcp_1 = __frcp_rn(sc_dec_13);
                        float inv_14 = ((sc_dec_13 > 0.0f) ? _rcp_1 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_1 = {inv_14, inv_14};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2)[_ls], _scale2_1);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv_2[_ls] = qv_2[_ls] * inv_14;
                        }
                        #endif
                        uint32_t _fp4_pair_8;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_8) : "f"(qv_2[0]), "f"(qv_2[1]));
                        uint32_t _fp4_pair_9;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_9) : "f"(qv_2[2]), "f"(qv_2[3]));
                        uint32_t _fp4_pair_10;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_10) : "f"(qv_2[4]), "f"(qv_2[5]));
                        uint32_t _fp4_pair_11;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_11) : "f"(qv_2[6]), "f"(qv_2[7]));
                        uint32_t _fp4_pair_12;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_12) : "f"(qv_2[8]), "f"(qv_2[9]));
                        uint32_t _fp4_pair_13;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_13) : "f"(qv_2[10]), "f"(qv_2[11]));
                        uint32_t _fp4_pair_14;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_14) : "f"(qv_2[12]), "f"(qv_2[13]));
                        uint32_t _fp4_pair_15;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_15) : "f"(qv_2[14]), "f"(qv_2[15]));
                        words[2] = _fp4_pair_8 | _fp4_pair_9 << 8 | _fp4_pair_10 << 16 | _fp4_pair_11 << 24;
                        words[3] = _fp4_pair_12 | _fp4_pair_13 << 8 | _fp4_pair_14 << 16 | _fp4_pair_15 << 24;
                        sf_word = sf_word | sc_byte_8 << 8;
                        unsigned int qa_15[4];
                        unsigned int qb_16[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15[(0) + 3]))
                            : "r"(q_row_addr + (4 ^ q_row % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16[(0) + 3]))
                            : "r"(q_row_addr + (5 ^ q_row % 8) * 16));
                        float qv_17[16];
                        qv_17[0] = __uint_as_float(qa_15[0] << 16);
                        qv_17[1] = __uint_as_float(qa_15[0] & 4294901760u);
                        qv_17[8] = __uint_as_float(qb_16[0] << 16);
                        qv_17[9] = __uint_as_float(qb_16[0] & 4294901760u);
                        qv_17[2] = __uint_as_float(qa_15[1] << 16);
                        qv_17[3] = __uint_as_float(qa_15[1] & 4294901760u);
                        qv_17[10] = __uint_as_float(qb_16[1] << 16);
                        qv_17[11] = __uint_as_float(qb_16[1] & 4294901760u);
                        qv_17[4] = __uint_as_float(qa_15[2] << 16);
                        qv_17[5] = __uint_as_float(qa_15[2] & 4294901760u);
                        qv_17[12] = __uint_as_float(qb_16[2] << 16);
                        qv_17[13] = __uint_as_float(qb_16[2] & 4294901760u);
                        qv_17[6] = __uint_as_float(qa_15[3] << 16);
                        qv_17[7] = __uint_as_float(qa_15[3] & 4294901760u);
                        qv_17[14] = __uint_as_float(qb_16[3] << 16);
                        qv_17[15] = __uint_as_float(qb_16[3] & 4294901760u);
                        float m8_18[8];
                        float _fabs_32 = fabsf(qv_17[0]);
                        float _fabs_33 = fabsf(qv_17[1]);
                        float _max_30 = max_noftz(_fabs_32, _fabs_33);
                        m8_18[0] = _max_30;
                        float _fabs_34 = fabsf(qv_17[2]);
                        float _fabs_35 = fabsf(qv_17[3]);
                        float _max_31 = max_noftz(_fabs_34, _fabs_35);
                        m8_18[1] = _max_31;
                        float _fabs_36 = fabsf(qv_17[4]);
                        float _fabs_37 = fabsf(qv_17[5]);
                        float _max_32 = max_noftz(_fabs_36, _fabs_37);
                        m8_18[2] = _max_32;
                        float _fabs_38 = fabsf(qv_17[6]);
                        float _fabs_39 = fabsf(qv_17[7]);
                        float _max_33 = max_noftz(_fabs_38, _fabs_39);
                        m8_18[3] = _max_33;
                        float _fabs_40 = fabsf(qv_17[8]);
                        float _fabs_41 = fabsf(qv_17[9]);
                        float _max_34 = max_noftz(_fabs_40, _fabs_41);
                        m8_18[4] = _max_34;
                        float _fabs_42 = fabsf(qv_17[10]);
                        float _fabs_43 = fabsf(qv_17[11]);
                        float _max_35 = max_noftz(_fabs_42, _fabs_43);
                        m8_18[5] = _max_35;
                        float _fabs_44 = fabsf(qv_17[12]);
                        float _fabs_45 = fabsf(qv_17[13]);
                        float _max_36 = max_noftz(_fabs_44, _fabs_45);
                        m8_18[6] = _max_36;
                        float _fabs_46 = fabsf(qv_17[14]);
                        float _fabs_47 = fabsf(qv_17[15]);
                        float _max_37 = max_noftz(_fabs_46, _fabs_47);
                        m8_18[7] = _max_37;
                        float m4_19[4];
                        float _max_38 = max_noftz(m8_18[0], m8_18[1]);
                        m4_19[0] = _max_38;
                        float _max_39 = max_noftz(m8_18[2], m8_18[3]);
                        m4_19[1] = _max_39;
                        float _max_40 = max_noftz(m8_18[4], m8_18[5]);
                        m4_19[2] = _max_40;
                        float _max_41 = max_noftz(m8_18[6], m8_18[7]);
                        m4_19[3] = _max_41;
                        float _max_42 = max_noftz(m4_19[0], m4_19[1]);
                        float _max_43 = max_noftz(m4_19[2], m4_19[3]);
                        float _max_44 = max_noftz(_max_42, _max_43);
                        float amax_20 = _max_44;
                        float sc_21 = amax_20 * inv_six;
                        uint16_t _e4m3x2_f32_2;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(0.0f), "f"(sc_21));
                        uint16_t sc_pair_22 = _e4m3x2_f32_2;
                        unsigned int sc_byte_23 = (unsigned int)sc_pair_22 & 255;
                        unsigned int sc_exp_24 = sc_byte_23 >> 3 & 15;
                        unsigned int sc_man_25 = sc_byte_23 & 7;
                        float sc_norm_26 = __uint_as_float(sc_exp_24 + 120 << 23 | sc_man_25 << 20);
                        float sc_sub_27 = (float)sc_man_25 * 0.001953125f;
                        float sc_dec_28 = ((sc_exp_24 == 0) ? sc_sub_27 : sc_norm_26);
                        float _rcp_2 = __frcp_rn(sc_dec_28);
                        float inv_29 = ((sc_dec_28 > 0.0f) ? _rcp_2 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_2 = {inv_29, inv_29};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_17)[_ls], _scale2_2);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv_17[_ls] = qv_17[_ls] * inv_29;
                        }
                        #endif
                        uint32_t _fp4_pair_16;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_16) : "f"(qv_17[0]), "f"(qv_17[1]));
                        uint32_t _fp4_pair_17;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_17) : "f"(qv_17[2]), "f"(qv_17[3]));
                        uint32_t _fp4_pair_18;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_18) : "f"(qv_17[4]), "f"(qv_17[5]));
                        uint32_t _fp4_pair_19;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_19) : "f"(qv_17[6]), "f"(qv_17[7]));
                        uint32_t _fp4_pair_20;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_20) : "f"(qv_17[8]), "f"(qv_17[9]));
                        uint32_t _fp4_pair_21;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_21) : "f"(qv_17[10]), "f"(qv_17[11]));
                        uint32_t _fp4_pair_22;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_22) : "f"(qv_17[12]), "f"(qv_17[13]));
                        uint32_t _fp4_pair_23;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_23) : "f"(qv_17[14]), "f"(qv_17[15]));
                        words[4] = _fp4_pair_16 | _fp4_pair_17 << 8 | _fp4_pair_18 << 16 | _fp4_pair_19 << 24;
                        words[5] = _fp4_pair_20 | _fp4_pair_21 << 8 | _fp4_pair_22 << 16 | _fp4_pair_23 << 24;
                        sf_word = sf_word | sc_byte_23 << 16;
                        unsigned int qa_30[4];
                        unsigned int qb_31[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30[(0) + 3]))
                            : "r"(q_row_addr + (6 ^ q_row % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31[(0) + 3]))
                            : "r"(q_row_addr + (7 ^ q_row % 8) * 16));
                        float qv_32[16];
                        qv_32[0] = __uint_as_float(qa_30[0] << 16);
                        qv_32[1] = __uint_as_float(qa_30[0] & 4294901760u);
                        qv_32[8] = __uint_as_float(qb_31[0] << 16);
                        qv_32[9] = __uint_as_float(qb_31[0] & 4294901760u);
                        qv_32[2] = __uint_as_float(qa_30[1] << 16);
                        qv_32[3] = __uint_as_float(qa_30[1] & 4294901760u);
                        qv_32[10] = __uint_as_float(qb_31[1] << 16);
                        qv_32[11] = __uint_as_float(qb_31[1] & 4294901760u);
                        qv_32[4] = __uint_as_float(qa_30[2] << 16);
                        qv_32[5] = __uint_as_float(qa_30[2] & 4294901760u);
                        qv_32[12] = __uint_as_float(qb_31[2] << 16);
                        qv_32[13] = __uint_as_float(qb_31[2] & 4294901760u);
                        qv_32[6] = __uint_as_float(qa_30[3] << 16);
                        qv_32[7] = __uint_as_float(qa_30[3] & 4294901760u);
                        qv_32[14] = __uint_as_float(qb_31[3] << 16);
                        qv_32[15] = __uint_as_float(qb_31[3] & 4294901760u);
                        float m8_33[8];
                        float _fabs_48 = fabsf(qv_32[0]);
                        float _fabs_49 = fabsf(qv_32[1]);
                        float _max_45 = max_noftz(_fabs_48, _fabs_49);
                        m8_33[0] = _max_45;
                        float _fabs_50 = fabsf(qv_32[2]);
                        float _fabs_51 = fabsf(qv_32[3]);
                        float _max_46 = max_noftz(_fabs_50, _fabs_51);
                        m8_33[1] = _max_46;
                        float _fabs_52 = fabsf(qv_32[4]);
                        float _fabs_53 = fabsf(qv_32[5]);
                        float _max_47 = max_noftz(_fabs_52, _fabs_53);
                        m8_33[2] = _max_47;
                        float _fabs_54 = fabsf(qv_32[6]);
                        float _fabs_55 = fabsf(qv_32[7]);
                        float _max_48 = max_noftz(_fabs_54, _fabs_55);
                        m8_33[3] = _max_48;
                        float _fabs_56 = fabsf(qv_32[8]);
                        float _fabs_57 = fabsf(qv_32[9]);
                        float _max_49 = max_noftz(_fabs_56, _fabs_57);
                        m8_33[4] = _max_49;
                        float _fabs_58 = fabsf(qv_32[10]);
                        float _fabs_59 = fabsf(qv_32[11]);
                        float _max_50 = max_noftz(_fabs_58, _fabs_59);
                        m8_33[5] = _max_50;
                        float _fabs_60 = fabsf(qv_32[12]);
                        float _fabs_61 = fabsf(qv_32[13]);
                        float _max_51 = max_noftz(_fabs_60, _fabs_61);
                        m8_33[6] = _max_51;
                        float _fabs_62 = fabsf(qv_32[14]);
                        float _fabs_63 = fabsf(qv_32[15]);
                        float _max_52 = max_noftz(_fabs_62, _fabs_63);
                        m8_33[7] = _max_52;
                        float m4_34[4];
                        float _max_53 = max_noftz(m8_33[0], m8_33[1]);
                        m4_34[0] = _max_53;
                        float _max_54 = max_noftz(m8_33[2], m8_33[3]);
                        m4_34[1] = _max_54;
                        float _max_55 = max_noftz(m8_33[4], m8_33[5]);
                        m4_34[2] = _max_55;
                        float _max_56 = max_noftz(m8_33[6], m8_33[7]);
                        m4_34[3] = _max_56;
                        float _max_57 = max_noftz(m4_34[0], m4_34[1]);
                        float _max_58 = max_noftz(m4_34[2], m4_34[3]);
                        float _max_59 = max_noftz(_max_57, _max_58);
                        float amax_35 = _max_59;
                        float sc_36 = amax_35 * inv_six;
                        uint16_t _e4m3x2_f32_3;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(0.0f), "f"(sc_36));
                        uint16_t sc_pair_37 = _e4m3x2_f32_3;
                        unsigned int sc_byte_38 = (unsigned int)sc_pair_37 & 255;
                        unsigned int sc_exp_39 = sc_byte_38 >> 3 & 15;
                        unsigned int sc_man_40 = sc_byte_38 & 7;
                        float sc_norm_41 = __uint_as_float(sc_exp_39 + 120 << 23 | sc_man_40 << 20);
                        float sc_sub_42 = (float)sc_man_40 * 0.001953125f;
                        float sc_dec_43 = ((sc_exp_39 == 0) ? sc_sub_42 : sc_norm_41);
                        float _rcp_3 = __frcp_rn(sc_dec_43);
                        float inv_44 = ((sc_dec_43 > 0.0f) ? _rcp_3 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_3 = {inv_44, inv_44};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_32)[_ls], _scale2_3);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv_32[_ls] = qv_32[_ls] * inv_44;
                        }
                        #endif
                        uint32_t _fp4_pair_24;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_24) : "f"(qv_32[0]), "f"(qv_32[1]));
                        uint32_t _fp4_pair_25;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_25) : "f"(qv_32[2]), "f"(qv_32[3]));
                        uint32_t _fp4_pair_26;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_26) : "f"(qv_32[4]), "f"(qv_32[5]));
                        uint32_t _fp4_pair_27;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_27) : "f"(qv_32[6]), "f"(qv_32[7]));
                        uint32_t _fp4_pair_28;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_28) : "f"(qv_32[8]), "f"(qv_32[9]));
                        uint32_t _fp4_pair_29;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_29) : "f"(qv_32[10]), "f"(qv_32[11]));
                        uint32_t _fp4_pair_30;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_30) : "f"(qv_32[12]), "f"(qv_32[13]));
                        uint32_t _fp4_pair_31;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_31) : "f"(qv_32[14]), "f"(qv_32[15]));
                        words[6] = _fp4_pair_24 | _fp4_pair_25 << 8 | _fp4_pair_26 << 16 | _fp4_pair_27 << 24;
                        words[7] = _fp4_pair_28 | _fp4_pair_29 << 8 | _fp4_pair_30 << 16 | _fp4_pair_31 << 24;
                        sf_word = sf_word | sc_byte_38 << 24;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_qf4_addr + (unsigned int)(2 * kset / 8 * 16384 + (q_row * 128 + (2 * kset % 8 * 16 ^ q_row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words[0])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_qf4_addr + (unsigned int)((2 * kset + 1) / 8 * 16384 + (q_row * 128 + ((2 * kset + 1) % 8 * 16 ^ q_row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words[4])), "r"(*reinterpret_cast<uint32_t*>(&words[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(4) + 3])));
                        smem_qsf32[kset / 4 * 2048 + q_row % 32 / 8 * 512 + kset % 4 * 128 + q_row % 8 * 16 + q_row / 32 % 4 * 4 >> 2] = sf_word;
                    } else {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(2 * kset / 8 * 16384 + (q_row * 128 + (2 * kset % 8 * 16 ^ q_row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)((2 * kset + 1) / 8 * 16384 + (q_row * 128 + ((2 * kset + 1) % 8 * 16 ^ q_row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        smem_qsf32[kset / 4 * 2048 + q_row % 32 / 8 * 512 + kset % 4 * 128 + q_row % 8 * 16 + q_row / 32 % 4 * 4 >> 2] = 0;
                    }
                }
                if (i == 0) {
                    mbarrier_arrive(q_ready_k_addr);
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            unsigned int _phase_q_ready_0 = 0;
            mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0, 10000000);
            _phase_q_ready_0 ^= 1;
            float m_run = -CAKE_INF;
            float l_run = 0.0f;
            float psum_run = 0.0f;
            unsigned int mask_words[4];
            int last_buf = 0;
            int last_par = 0;
            float sv0[16];
            float sv1[16];
            float sv2[16];
            unsigned int packed_p[4];
            unsigned int chunk_bits = 0;
            float ov[16];
            int o_addr = 0;
            float psum_v[1];
            mbarrier_wait_hint(tile_meta_addr, 0, 10000000);
            unsigned int my_mask = smem_mask[local_warp];
            int valid = (int)(my_mask >> (unsigned int)lane & 1);
            mbarrier_wait_cluster_hint(kv_landed_addr, 0, 10000000);
            int kbase = smem_v5_addr;
            int kzone_off = 0;
            int vbase = smem_v_addr;
            unsigned int zero4[4];
            zero4[0] = 0;
            zero4[1] = 0;
            zero4[2] = 0;
            zero4[3] = 0;
            int znope_hi_v = 8;
            int zrope_pairs_hi_v = 0;
            int nope_lo = ((rank == 0) ? 0 : 8);
            int nope_hi_v = ((rank == 0) ? 3 : 12);
            int rope_pair_lo = 0;
            int rope_pairs_hi_v = 0;
            int vrank8 = 8 * rank;
            if (valid != 0) {
                #pragma unroll 1
                for (int c = nope_lo; c < nope_hi_v; c++) {
                    unsigned int raw[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 3]))
                        : "r"(kbase + (c / 8 * 16384 + (row * 128 + (c % 8 * 16 ^ row % 8 * 16)))));
                    int sf_w = c / 2;
                    unsigned int sf_word_1 = smem_kzone32[kzone_off + ((14 + (sf_w >> 2)) / 8 * 16384 + (row * 128 + ((14 + (sf_w >> 2)) % 8 * 16 ^ row % 8 * 16))) + 4 * (sf_w & 3) >> 2];
                    int block = 2 * c;
                    unsigned int scale = sf_word_1 >> (unsigned int)(8 * ((c & 1) * 2)) & 255;
                    unsigned int v8[4];
                    {
                        v8[0] = cake_dsv4_qmul4_portable<5>(raw[0], scale);
                    }
                    {
                        v8[1] = cake_dsv4_qmul4_portable<6>(raw[0], scale);
                    }
                    {
                        v8[2] = cake_dsv4_qmul4_portable<5>(raw[1], scale);
                    }
                    {
                        v8[3] = cake_dsv4_qmul4_portable<6>(raw[1], scale);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase + (c - vrank8) / 4 * 16384 + (row * 128 + (2 * c % 8 * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8[0])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 3])));
                    int block_0 = 2 * c + 1;
                    unsigned int scale_1 = sf_word_1 >> (unsigned int)(8 * ((c & 1) * 2 + 1)) & 255;
                    unsigned int v8_2[4];
                    {
                        v8_2[0] = cake_dsv4_qmul4_portable<5>(raw[2], scale_1);
                    }
                    {
                        v8_2[1] = cake_dsv4_qmul4_portable<6>(raw[2], scale_1);
                    }
                    {
                        v8_2[2] = cake_dsv4_qmul4_portable<5>(raw[3], scale_1);
                    }
                    {
                        v8_2[3] = cake_dsv4_qmul4_portable<6>(raw[3], scale_1);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase + (c - vrank8) / 4 * 16384 + (row * 128 + ((2 * c + 1) % 8 * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 3])));
                }
                for (int jp = rope_pair_lo; jp < rope_pairs_hi_v; jp++) {
                    unsigned int vrope[4];
                    int j = 2 * jp;
                    unsigned int rope[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 3]))
                        : "r"(kbase + 36864 + (row * 128 + (j * 16 ^ row % 8 * 16))));
                    float lo = __uint_as_float(rope[0] << 16);
                    float hi = __uint_as_float(rope[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_4;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_4) : "f"(hi), "f"(lo));
                    uint16_t pair = _e4m3x2_f32_4;
                    {
                        vrope[0] = (unsigned int)pair;
                    }
                    float lo_0 = __uint_as_float(rope[1] << 16);
                    float hi_1 = __uint_as_float(rope[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_5;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_5) : "f"(hi_1), "f"(lo_0));
                    uint16_t pair_2 = _e4m3x2_f32_5;
                    {
                        vrope[0] = vrope[0] | (unsigned int)pair_2 << 16;
                    }
                    float lo_3 = __uint_as_float(rope[2] << 16);
                    float hi_4 = __uint_as_float(rope[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_6;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(hi_4), "f"(lo_3));
                    uint16_t pair_5 = _e4m3x2_f32_6;
                    {
                        vrope[1] = (unsigned int)pair_5;
                    }
                    float lo_6 = __uint_as_float(rope[3] << 16);
                    float hi_7 = __uint_as_float(rope[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_7;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_7) : "f"(hi_7), "f"(lo_6));
                    uint16_t pair_8 = _e4m3x2_f32_7;
                    {
                        vrope[1] = vrope[1] | (unsigned int)pair_8 << 16;
                    }
                    int j_9 = 2 * jp + 1;
                    unsigned int rope_10[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10[(0) + 3]))
                        : "r"(kbase + 36864 + (row * 128 + (j_9 * 16 ^ row % 8 * 16))));
                    float lo_11 = __uint_as_float(rope_10[0] << 16);
                    float hi_12 = __uint_as_float(rope_10[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_8;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_8) : "f"(hi_12), "f"(lo_11));
                    uint16_t pair_13 = _e4m3x2_f32_8;
                    {
                        vrope[2] = (unsigned int)pair_13;
                    }
                    float lo_14 = __uint_as_float(rope_10[1] << 16);
                    float hi_15 = __uint_as_float(rope_10[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_9;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_9) : "f"(hi_15), "f"(lo_14));
                    uint16_t pair_16 = _e4m3x2_f32_9;
                    {
                        vrope[2] = vrope[2] | (unsigned int)pair_16 << 16;
                    }
                    float lo_17 = __uint_as_float(rope_10[2] << 16);
                    float hi_18 = __uint_as_float(rope_10[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_10;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_10) : "f"(hi_18), "f"(lo_17));
                    uint16_t pair_19 = _e4m3x2_f32_10;
                    {
                        vrope[3] = (unsigned int)pair_19;
                    }
                    float lo_20 = __uint_as_float(rope_10[3] << 16);
                    float hi_21 = __uint_as_float(rope_10[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_11;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_11) : "f"(hi_21), "f"(lo_20));
                    uint16_t pair_22 = _e4m3x2_f32_11;
                    {
                        vrope[3] = vrope[3] | (unsigned int)pair_22 << 16;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase + 16384 + (row * 128 + ((4 + jp) * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&vrope[0])), "r"(*reinterpret_cast<uint32_t*>(&vrope[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vrope[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vrope[(0) + 3])));
                }
            } else {
                {
                    for (int c_1 = 0; c_1 < znope_hi_v; c_1++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(kbase + (c_1 / 8 * 16384 + (row * 128 + (c_1 % 8 * 16 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&zero4[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 3])));
                    }
                }
                for (int c_2 = nope_lo; c_2 < nope_hi_v; c_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase + (c_2 - vrank8) / 4 * 16384 + (row * 128 + (2 * c_2 % 8 * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase + (c_2 - vrank8) / 4 * 16384 + (row * 128 + ((2 * c_2 + 1) % 8 * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 3])));
                }
                for (int jp_1 = rope_pair_lo; jp_1 < rope_pairs_hi_v; jp_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase + 16384 + (row * 128 + ((4 + jp_1) * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 3])));
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            #pragma unroll 1
            for (int t = tile_lo; t < tile_hi; t++) {
                int it = t - tile_lo;
                int buf = it & 1;
                int par = it >> 1 & 1;
                if (it > 0) {
                    mbarrier_wait_hint(pv_done_addr + (last_buf) * 8, last_par, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(kbuf_free_p_addr + (last_buf) * 8);
                }
                mask_words[0] = smem_mask[(it & 3) * 4];
                mask_words[1] = smem_mask[(it & 3) * 4 + 1];
                mask_words[2] = smem_mask[(it & 3) * 4 + 2];
                mask_words[3] = smem_mask[(it & 3) * 4 + 3];
                kbase = smem_v5_addr + (unsigned int)(buf * 118784);
                mbarrier_wait_hint(s_full_addr + (buf) * 8, par, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp_rows_valid != 0) {
                    tmem_ld_x16(&sv0[0], taddr + 256 + (unsigned int)(tmem_row_origin << 16));
                    tmem_ld_x16(&sv1[0], taddr + 256 + 16 + (unsigned int)(tmem_row_origin << 16));
                    tmem_ld_x16(&sv2[0], taddr + 256 + 32 + (unsigned int)(tmem_row_origin << 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                }
                mbarrier_arrive(s_free_addr);
                mbarrier_arrive(kbuf_free_k_addr + (buf) * 8);
                float slice_max = -CAKE_INF;
                if (warp_rows_valid != 0) {
                    chunk_bits = mask_words[0] & 65535;
                    if (chunk_bits != 65535) {
                        if ((chunk_bits & 1) == 0) {
                            sv0[0] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 1 & 1) == 0) {
                            sv0[1] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 2 & 1) == 0) {
                            sv0[2] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 3 & 1) == 0) {
                            sv0[3] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 4 & 1) == 0) {
                            sv0[4] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 5 & 1) == 0) {
                            sv0[5] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 6 & 1) == 0) {
                            sv0[6] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 7 & 1) == 0) {
                            sv0[7] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 8 & 1) == 0) {
                            sv0[8] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 9 & 1) == 0) {
                            sv0[9] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 10 & 1) == 0) {
                            sv0[10] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 11 & 1) == 0) {
                            sv0[11] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 12 & 1) == 0) {
                            sv0[12] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 13 & 1) == 0) {
                            sv0[13] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 14 & 1) == 0) {
                            sv0[14] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 15 & 1) == 0) {
                            sv0[15] = -CAKE_INF;
                        }
                    }
                    float sv0_max = sv0[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv0_max = max_noftz(sv0_max, sv0[_lr]);
                    }
                    float _max_60 = max_noftz(slice_max, sv0_max);
                    slice_max = _max_60;
                    chunk_bits = mask_words[0] >> 16 & 65535;
                    if (chunk_bits != 65535) {
                        if ((chunk_bits & 1) == 0) {
                            sv1[0] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 1 & 1) == 0) {
                            sv1[1] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 2 & 1) == 0) {
                            sv1[2] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 3 & 1) == 0) {
                            sv1[3] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 4 & 1) == 0) {
                            sv1[4] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 5 & 1) == 0) {
                            sv1[5] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 6 & 1) == 0) {
                            sv1[6] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 7 & 1) == 0) {
                            sv1[7] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 8 & 1) == 0) {
                            sv1[8] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 9 & 1) == 0) {
                            sv1[9] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 10 & 1) == 0) {
                            sv1[10] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 11 & 1) == 0) {
                            sv1[11] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 12 & 1) == 0) {
                            sv1[12] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 13 & 1) == 0) {
                            sv1[13] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 14 & 1) == 0) {
                            sv1[14] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 15 & 1) == 0) {
                            sv1[15] = -CAKE_INF;
                        }
                    }
                    float sv1_max = sv1[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv1_max = max_noftz(sv1_max, sv1[_lr]);
                    }
                    float _max_61 = max_noftz(slice_max, sv1_max);
                    slice_max = _max_61;
                    chunk_bits = mask_words[1] & 65535;
                    if (chunk_bits != 65535) {
                        if ((chunk_bits & 1) == 0) {
                            sv2[0] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 1 & 1) == 0) {
                            sv2[1] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 2 & 1) == 0) {
                            sv2[2] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 3 & 1) == 0) {
                            sv2[3] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 4 & 1) == 0) {
                            sv2[4] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 5 & 1) == 0) {
                            sv2[5] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 6 & 1) == 0) {
                            sv2[6] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 7 & 1) == 0) {
                            sv2[7] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 8 & 1) == 0) {
                            sv2[8] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 9 & 1) == 0) {
                            sv2[9] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 10 & 1) == 0) {
                            sv2[10] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 11 & 1) == 0) {
                            sv2[11] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 12 & 1) == 0) {
                            sv2[12] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 13 & 1) == 0) {
                            sv2[13] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 14 & 1) == 0) {
                            sv2[14] = -CAKE_INF;
                        }
                        if ((chunk_bits >> 15 & 1) == 0) {
                            sv2[15] = -CAKE_INF;
                        }
                    }
                    float sv2_max = sv2[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv2_max = max_noftz(sv2_max, sv2[_lr]);
                    }
                    float _max_62 = max_noftz(slice_max, sv2_max);
                    slice_max = _max_62;
                }
                smem_pmax[row] = slice_max;
                asm volatile("barrier.sync 8, 384;" ::: "memory");
                float _max_63 = max_noftz(smem_pmax[row], smem_pmax[128 + row]);
                float _max_64 = max_noftz(_max_63, smem_pmax[256 + row]);
                float tile_max = _max_64;
                float cand = tile_max * softmax_scale_log2;
                if (it == 0) {
                    if (has_sink_row != 0) {
                        float _max_65 = max_noftz(cand, sink_log2);
                        cand = _max_65;
                    }
                }
                float _max_66 = max_noftz(cand, m_run);
                cand = _max_66;
                int grow = 0;
                if (it == 0) {
                    grow = 1;
                }
                if (cand - m_run > 8.0f) {
                    grow = 1;
                }
                if (grow != 0) {
                    float _exp2_0 = approx_exp2(m_run - cand);
                    float alpha = ((m_run > -CAKE_INF) ? _exp2_0 : 0.0f);
                    l_run = l_run * alpha;
                    psum_run = psum_run * alpha;
                    if (it > 0) {
                        if (warp_rows_valid != 0) {
                            o_addr = taddr + (unsigned int)(tmem_row_origin << 16);
                            tmem_ld_x16(&ov[0], o_addr);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_12 = {alpha, alpha};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov)[_ls], _scale2_12);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov[_ls] = ov[_ls] * alpha;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr, ov);
                            o_addr = taddr + 16 + (unsigned int)(tmem_row_origin << 16);
                            tmem_ld_x16(&ov[0], o_addr);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_13 = {alpha, alpha};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov)[_ls], _scale2_13);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov[_ls] = ov[_ls] * alpha;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr, ov);
                            o_addr = taddr + 32 + (unsigned int)(tmem_row_origin << 16);
                            tmem_ld_x16(&ov[0], o_addr);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_14 = {alpha, alpha};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov)[_ls], _scale2_14);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov[_ls] = ov[_ls] * alpha;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr, ov);
                            o_addr = taddr + 48 + (unsigned int)(tmem_row_origin << 16);
                            tmem_ld_x16(&ov[0], o_addr);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_15 = {alpha, alpha};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov)[_ls], _scale2_15);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov[_ls] = ov[_ls] * alpha;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr, ov);
                            o_addr = taddr + 64 + (unsigned int)(tmem_row_origin << 16);
                            tmem_ld_x16(&ov[0], o_addr);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_16 = {alpha, alpha};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov)[_ls], _scale2_16);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov[_ls] = ov[_ls] * alpha;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr, ov);
                            o_addr = taddr + 80 + (unsigned int)(tmem_row_origin << 16);
                            tmem_ld_x16(&ov[0], o_addr);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_17 = {alpha, alpha};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov)[_ls], _scale2_17);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov[_ls] = ov[_ls] * alpha;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr, ov);
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                    }
                    m_run = cand;
                }
                float m_use = ((m_run > -CAKE_INF) ? m_run : 0.0f);
                float slice_sum = 0.0f;
                if (warp_rows_valid != 0) {
                    float score_bias = -m_use;
                    const float2 _fma_b2_18 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_19 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv0)[_lf], _fma_b2_18, _fma_c2_19);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv0[_le] = approx_exp2(sv0[_le]);
                    }
                    float sv0_sum = 0.0f;
                    #pragma unroll
                    for (int _lr = 0; _lr < 16; _lr++) {
                        sv0_sum += sv0[_lr];
                    }
                    slice_sum = slice_sum + sv0_sum;
                    if (row_valid == 0) {
                        sv0[0] = 0.0f;
                        sv0[1] = 0.0f;
                        sv0[2] = 0.0f;
                        sv0[3] = 0.0f;
                        sv0[4] = 0.0f;
                        sv0[5] = 0.0f;
                        sv0[6] = 0.0f;
                        sv0[7] = 0.0f;
                        sv0[8] = 0.0f;
                        sv0[9] = 0.0f;
                        sv0[10] = 0.0f;
                        sv0[11] = 0.0f;
                        sv0[12] = 0.0f;
                        sv0[13] = 0.0f;
                        sv0[14] = 0.0f;
                        sv0[15] = 0.0f;
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
                            : "=r"(_packed) : "f"(sv0[0]), "f"(sv0[1]),
                                               "f"(sv0[2]), "f"(sv0[3]));
                        packed_p[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv0[4]), "f"(sv0[5]),
                                               "f"(sv0[6]), "f"(sv0[7]));
                        packed_p[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv0[8]), "f"(sv0[9]),
                                               "f"(sv0[10]), "f"(sv0[11]));
                        packed_p[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv0[12]), "f"(sv0[13]),
                                               "f"(sv0[14]), "f"(sv0[15]));
                        packed_p[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(kbase + 36864 + (row * 128 + (0 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 3])));
                    const float2 _fma_b2_20 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_21 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv1)[_lf], _fma_b2_20, _fma_c2_21);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv1[_le] = approx_exp2(sv1[_le]);
                    }
                    float sv1_sum = 0.0f;
                    #pragma unroll
                    for (int _lr = 0; _lr < 16; _lr++) {
                        sv1_sum += sv1[_lr];
                    }
                    slice_sum = slice_sum + sv1_sum;
                    if (row_valid == 0) {
                        sv1[0] = 0.0f;
                        sv1[1] = 0.0f;
                        sv1[2] = 0.0f;
                        sv1[3] = 0.0f;
                        sv1[4] = 0.0f;
                        sv1[5] = 0.0f;
                        sv1[6] = 0.0f;
                        sv1[7] = 0.0f;
                        sv1[8] = 0.0f;
                        sv1[9] = 0.0f;
                        sv1[10] = 0.0f;
                        sv1[11] = 0.0f;
                        sv1[12] = 0.0f;
                        sv1[13] = 0.0f;
                        sv1[14] = 0.0f;
                        sv1[15] = 0.0f;
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
                            : "=r"(_packed) : "f"(sv1[0]), "f"(sv1[1]),
                                               "f"(sv1[2]), "f"(sv1[3]));
                        packed_p[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv1[4]), "f"(sv1[5]),
                                               "f"(sv1[6]), "f"(sv1[7]));
                        packed_p[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv1[8]), "f"(sv1[9]),
                                               "f"(sv1[10]), "f"(sv1[11]));
                        packed_p[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv1[12]), "f"(sv1[13]),
                                               "f"(sv1[14]), "f"(sv1[15]));
                        packed_p[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(kbase + 36864 + (row * 128 + (16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 3])));
                    const float2 _fma_b2_22 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_23 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv2)[_lf], _fma_b2_22, _fma_c2_23);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv2[_le] = approx_exp2(sv2[_le]);
                    }
                    float sv2_sum = 0.0f;
                    #pragma unroll
                    for (int _lr = 0; _lr < 16; _lr++) {
                        sv2_sum += sv2[_lr];
                    }
                    slice_sum = slice_sum + sv2_sum;
                    if (row_valid == 0) {
                        sv2[0] = 0.0f;
                        sv2[1] = 0.0f;
                        sv2[2] = 0.0f;
                        sv2[3] = 0.0f;
                        sv2[4] = 0.0f;
                        sv2[5] = 0.0f;
                        sv2[6] = 0.0f;
                        sv2[7] = 0.0f;
                        sv2[8] = 0.0f;
                        sv2[9] = 0.0f;
                        sv2[10] = 0.0f;
                        sv2[11] = 0.0f;
                        sv2[12] = 0.0f;
                        sv2[13] = 0.0f;
                        sv2[14] = 0.0f;
                        sv2[15] = 0.0f;
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
                            : "=r"(_packed) : "f"(sv2[0]), "f"(sv2[1]),
                                               "f"(sv2[2]), "f"(sv2[3]));
                        packed_p[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv2[4]), "f"(sv2[5]),
                                               "f"(sv2[6]), "f"(sv2[7]));
                        packed_p[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv2[8]), "f"(sv2[9]),
                                               "f"(sv2[10]), "f"(sv2[11]));
                        packed_p[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv2[12]), "f"(sv2[13]),
                                               "f"(sv2[14]), "f"(sv2[15]));
                        packed_p[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(kbase + 36864 + (row * 128 + (32 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 3])));
                }
                smem_psum[row] = slice_sum;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr + (buf) * 8);
                asm volatile("barrier.sync 8, 384;" ::: "memory");
                l_run = l_run + smem_psum[row] + smem_psum[128 + row] + smem_psum[256 + row];
                mbarrier_wait_hint(psum_full_addr + (buf) * 8, par, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp_rows_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                        " {%0}, [%1];"
                        : "=f"(psum_v[0])
                        : "r"(taddr + 448 + (unsigned int)(tmem_row_origin << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    psum_run = psum_run + psum_v[0];
                }
                if (tile_hi > t + 1) {
                    int nbuf = buf ^ 1;
                    int npar = it + 1 >> 1 & 1;
                    mbarrier_wait_hint(tile_meta_addr + (it + 1 & 3) * 8, it + 1 >> 2 & 1, 10000000);
                    my_mask = smem_mask[(it + 1 & 3) * 4 + local_warp];
                    valid = (int)(my_mask >> (unsigned int)lane & 1);
                    mbarrier_wait_cluster_hint(kv_landed_addr + (nbuf) * 8, npar, 10000000);
                    kbase = smem_v5_addr + (unsigned int)(nbuf * 118784);
                    kzone_off = nbuf * 118784;
                    vbase = smem_v_addr + (unsigned int)(nbuf * 32768);
                    unsigned int zero4_0[4];
                    zero4_0[0] = 0;
                    zero4_0[1] = 0;
                    zero4_0[2] = 0;
                    zero4_0[3] = 0;
                    int znope_hi_v_1 = 8;
                    int zrope_pairs_hi_v_2 = 0;
                    int nope_lo_3 = ((rank == 0) ? 0 : 8);
                    int nope_hi_v_4 = ((rank == 0) ? 3 : 12);
                    int rope_pair_lo_5 = 0;
                    int rope_pairs_hi_v_6 = 0;
                    int vrank8_7 = 8 * rank;
                    if (valid != 0) {
                        #pragma unroll 1
                        for (int c_3 = nope_lo_3; c_3 < nope_hi_v_4; c_3++) {
                            unsigned int raw_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 3]))
                                : "r"(kbase + (c_3 / 8 * 16384 + (row * 128 + (c_3 % 8 * 16 ^ row % 8 * 16)))));
                            int sf_w_1 = c_3 / 2;
                            unsigned int sf_word_2 = smem_kzone32[kzone_off + ((14 + (sf_w_1 >> 2)) / 8 * 16384 + (row * 128 + ((14 + (sf_w_1 >> 2)) % 8 * 16 ^ row % 8 * 16))) + 4 * (sf_w_1 & 3) >> 2];
                            int block_1 = 2 * c_3;
                            unsigned int scale_2 = sf_word_2 >> (unsigned int)(8 * ((c_3 & 1) * 2)) & 255;
                            unsigned int v8_1[4];
                            {
                                v8_1[0] = cake_dsv4_qmul4_portable<5>(raw_1[0], scale_2);
                            }
                            {
                                v8_1[1] = cake_dsv4_qmul4_portable<6>(raw_1[0], scale_2);
                            }
                            {
                                v8_1[2] = cake_dsv4_qmul4_portable<5>(raw_1[1], scale_2);
                            }
                            {
                                v8_1[3] = cake_dsv4_qmul4_portable<6>(raw_1[1], scale_2);
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase + (c_3 - vrank8_7) / 4 * 16384 + (row * 128 + (2 * c_3 % 8 * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 3])));
                            int block_0_1 = 2 * c_3 + 1;
                            unsigned int scale_1_1 = sf_word_2 >> (unsigned int)(8 * ((c_3 & 1) * 2 + 1)) & 255;
                            unsigned int v8_2_1[4];
                            {
                                v8_2_1[0] = cake_dsv4_qmul4_portable<5>(raw_1[2], scale_1_1);
                            }
                            {
                                v8_2_1[1] = cake_dsv4_qmul4_portable<6>(raw_1[2], scale_1_1);
                            }
                            {
                                v8_2_1[2] = cake_dsv4_qmul4_portable<5>(raw_1[3], scale_1_1);
                            }
                            {
                                v8_2_1[3] = cake_dsv4_qmul4_portable<6>(raw_1[3], scale_1_1);
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase + (c_3 - vrank8_7) / 4 * 16384 + (row * 128 + ((2 * c_3 + 1) % 8 * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_2_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_1[(0) + 3])));
                        }
                        for (int jp_2 = rope_pair_lo_5; jp_2 < rope_pairs_hi_v_6; jp_2++) {
                            unsigned int vrope_1[4];
                            int j_1 = 2 * jp_2;
                            unsigned int rope_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 3]))
                                : "r"(kbase + 36864 + (row * 128 + (j_1 * 16 ^ row % 8 * 16))));
                            float lo_1 = __uint_as_float(rope_1[0] << 16);
                            float hi_2 = __uint_as_float(rope_1[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_12;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_12) : "f"(hi_2), "f"(lo_1));
                            uint16_t pair_1 = _e4m3x2_f32_12;
                            {
                                vrope_1[0] = (unsigned int)pair_1;
                            }
                            float lo_0_1 = __uint_as_float(rope_1[1] << 16);
                            float hi_1_1 = __uint_as_float(rope_1[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_13;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_13) : "f"(hi_1_1), "f"(lo_0_1));
                            uint16_t pair_2_1 = _e4m3x2_f32_13;
                            {
                                vrope_1[0] = vrope_1[0] | (unsigned int)pair_2_1 << 16;
                            }
                            float lo_3_1 = __uint_as_float(rope_1[2] << 16);
                            float hi_4_1 = __uint_as_float(rope_1[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_14;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_14) : "f"(hi_4_1), "f"(lo_3_1));
                            uint16_t pair_5_1 = _e4m3x2_f32_14;
                            {
                                vrope_1[1] = (unsigned int)pair_5_1;
                            }
                            float lo_6_1 = __uint_as_float(rope_1[3] << 16);
                            float hi_7_1 = __uint_as_float(rope_1[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_15;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_15) : "f"(hi_7_1), "f"(lo_6_1));
                            uint16_t pair_8_1 = _e4m3x2_f32_15;
                            {
                                vrope_1[1] = vrope_1[1] | (unsigned int)pair_8_1 << 16;
                            }
                            int j_9_1 = 2 * jp_2 + 1;
                            unsigned int rope_10_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_10_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_1[(0) + 3]))
                                : "r"(kbase + 36864 + (row * 128 + (j_9_1 * 16 ^ row % 8 * 16))));
                            float lo_11_1 = __uint_as_float(rope_10_1[0] << 16);
                            float hi_12_1 = __uint_as_float(rope_10_1[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_16;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_16) : "f"(hi_12_1), "f"(lo_11_1));
                            uint16_t pair_13_1 = _e4m3x2_f32_16;
                            {
                                vrope_1[2] = (unsigned int)pair_13_1;
                            }
                            float lo_14_1 = __uint_as_float(rope_10_1[1] << 16);
                            float hi_15_1 = __uint_as_float(rope_10_1[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_17;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_17) : "f"(hi_15_1), "f"(lo_14_1));
                            uint16_t pair_16_1 = _e4m3x2_f32_17;
                            {
                                vrope_1[2] = vrope_1[2] | (unsigned int)pair_16_1 << 16;
                            }
                            float lo_17_1 = __uint_as_float(rope_10_1[2] << 16);
                            float hi_18_1 = __uint_as_float(rope_10_1[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_18;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_18) : "f"(hi_18_1), "f"(lo_17_1));
                            uint16_t pair_19_1 = _e4m3x2_f32_18;
                            {
                                vrope_1[3] = (unsigned int)pair_19_1;
                            }
                            float lo_20_1 = __uint_as_float(rope_10_1[3] << 16);
                            float hi_21_1 = __uint_as_float(rope_10_1[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_19;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_19) : "f"(hi_21_1), "f"(lo_20_1));
                            uint16_t pair_22_1 = _e4m3x2_f32_19;
                            {
                                vrope_1[3] = vrope_1[3] | (unsigned int)pair_22_1 << 16;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase + 16384 + (row * 128 + ((4 + jp_2) * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&vrope_1[0])), "r"(*reinterpret_cast<uint32_t*>(&vrope_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vrope_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vrope_1[(0) + 3])));
                        }
                    } else {
                        {
                            for (int c_4 = 0; c_4 < znope_hi_v_1; c_4++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                    "r"(kbase + (c_4 / 8 * 16384 + (row * 128 + (c_4 % 8 * 16 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 3])));
                            }
                        }
                        for (int c_5 = nope_lo_3; c_5 < nope_hi_v_4; c_5++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase + (c_5 - vrank8_7) / 4 * 16384 + (row * 128 + (2 * c_5 % 8 * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 3])));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase + (c_5 - vrank8_7) / 4 * 16384 + (row * 128 + ((2 * c_5 + 1) % 8 * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 3])));
                        }
                        for (int jp_3 = rope_pair_lo_5; jp_3 < rope_pairs_hi_v_6; jp_3++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase + 16384 + (row * 128 + ((4 + jp_3) * 16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0[(0) + 3])));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_full_addr + (nbuf) * 8);
                }
                last_buf = buf;
                last_par = par;
            }
            mbarrier_wait_hint(pv_done_addr + (last_buf) * 8, last_par, 10000000);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float sink_term = 0.0f;
            if (has_sink_row != 0) {
                float _exp2_1 = approx_exp2(sink_log2 - m_run);
                sink_term = _exp2_1;
            }
            if (row_valid != 0 && rank == 0) {
                float row_sum = l_run + sink_term;
                int lse_offset = (query_idx * num_heads + head_row) * num_splits + split_idx;
                float _log2_0;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(row_sum));
                partial_lse[lse_offset] = ((row_sum > 0.0f) ? (m_run + _log2_0) * lse_partial_scale : -CAKE_INF);
            }
            float denom = psum_run + sink_term;
            float _rcp_4 = approx_rcp(denom);
            float norm = ((denom > 0.0f) ? _rcp_4 * output_scale : 0.0f);
            float o_values[16];
            unsigned int packed_o[8];
            int stage_base = smem_v_addr;
            if (warp_rows_valid != 0) {
                tmem_ld_x16(&o_values[0], taddr + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_32 = {norm, norm};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values)[_ls], _scale2_32);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values[_ls] = o_values[_ls] * norm;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values[_lp*2 + 0], o_values[_lp*2+1 + 0]));
                    packed_o[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + (row * 128 + (0 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + (row * 128 + (16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 3])));
                tmem_ld_x16(&o_values[0], taddr + 16 + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_33 = {norm, norm};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values)[_ls], _scale2_33);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values[_ls] = o_values[_ls] * norm;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values[_lp*2 + 0], o_values[_lp*2+1 + 0]));
                    packed_o[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + (row * 128 + (32 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + (row * 128 + (48 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 3])));
                tmem_ld_x16(&o_values[0], taddr + 32 + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_34 = {norm, norm};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values)[_ls], _scale2_34);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values[_ls] = o_values[_ls] * norm;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values[_lp*2 + 0], o_values[_lp*2+1 + 0]));
                    packed_o[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + (row * 128 + (64 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + (row * 128 + (80 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 3])));
                tmem_ld_x16(&o_values[0], taddr + 48 + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_35 = {norm, norm};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values)[_ls], _scale2_35);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values[_ls] = o_values[_ls] * norm;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values[_lp*2 + 0], o_values[_lp*2+1 + 0]));
                    packed_o[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + (row * 128 + (96 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + (row * 128 + (112 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 3])));
                tmem_ld_x16(&o_values[0], taddr + 64 + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_36 = {norm, norm};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values)[_ls], _scale2_36);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values[_ls] = o_values[_ls] * norm;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values[_lp*2 + 0], o_values[_lp*2+1 + 0]));
                    packed_o[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + 16384 + (row * 128 + (0 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + 16384 + (row * 128 + (16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 3])));
                tmem_ld_x16(&o_values[0], taddr + 80 + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_37 = {norm, norm};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values)[_ls], _scale2_37);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values[_ls] = o_values[_ls] * norm;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values[_lp*2 + 0], o_values[_lp*2+1 + 0]));
                    packed_o[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + 16384 + (row * 128 + (32 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base + 16384 + (row * 128 + (48 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o[(4) + 3])));
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            if (row == 0) {
                tma_store_4d((&tmap_out), rank * 256, split_idx, head_base, query_idx, stage_base);
                tma_store_4d((&tmap_out), rank * 256 + 64, split_idx, head_base, query_idx, stage_base + 16384);
                tma_store_4d((&tmap_out), rank * 256 + 128, split_idx, head_base, query_idx, stage_base + 32768);
                tma_store_4d((&tmap_out), rank * 256 + 192, split_idx, head_base, query_idx, stage_base + 49152);
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            mbarrier_arrive(tmem_dealloc_addr);
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
    // ---- Role: compute1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // compute1_main
            const int local_warp_1 = warp - 4;
            int rank_1 = cta_rank;
            int work_idx_1 = blockIdx.x / 2;
            int head_tile_1 = work_idx_1 % num_head_tiles;
            int split_work_1 = work_idx_1 / num_head_tiles;
            int split_idx_1 = split_work_1 % num_splits;
            int query_idx_1 = split_work_1 / num_splits;
            int head_base_1 = head_tile_1 * 128;
            const int row_1 = local_warp_1 * 32 + lane;
            int head_row_1 = head_base_1 + row_1;
            int row_valid_1 = ((head_row_1 < num_heads) ? 1 : 0);
            int warp_rows_valid_1 = ((head_base_1 + local_warp_1 * 32 < num_heads) ? 1 : 0);
            const int tmem_row_origin_1 = local_warp_1 * 32;
            int tile_lo_1 = split_idx_1 * tiles_per_split;
            int tile_hi_1 = tile_lo_1 + tiles_per_split;
            if (tile_hi_1 > total_tiles) {
                tile_hi_1 = total_tiles;
            }
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_1 = bmm2_scale[0];
            float sink_log2_1 = 0.0f;
            int has_sink_row_1 = 0;
            if (has_sinks != 0 && split_idx_1 == 0 && row_valid_1 != 0) {
                has_sink_row_1 = 1;
                sink_log2_1 = sinks[head_row_1] * 1.4426950408889634f;
            }
            float inv_six_1 = 0.16666666666666666f;
            unsigned int _phase_q_nope_full0_0_1 = 0;
            mbarrier_wait_hint(q_nope_full0_addr, _phase_q_nope_full0_0_1, 10000000);
            _phase_q_nope_full0_0_1 ^= 1;
            unsigned int _phase_q_nope_full1_0_1 = 0;
            mbarrier_wait_hint(q_nope_full1_addr, _phase_q_nope_full1_0_1, 10000000);
            _phase_q_nope_full1_0_1 ^= 1;
            unsigned int _phase_q_nope_full2_0_1 = 0;
            mbarrier_wait_hint(q_nope_full2_addr, _phase_q_nope_full2_0_1, 10000000);
            _phase_q_nope_full2_0_1 ^= 1;
            const int q_warp_1 = warp;
            int unit_1 = 0;
            int q_block_1 = 0;
            int kset_1 = 0;
            int q_row_1 = 0;
            int q_row_addr_1 = 0;
            for (int i_1 = 0; i_1 < 3; i_1++) {
                unit_1 = q_warp_1 + 12 * i_1;
                if (unit_1 < 28) {
                    q_block_1 = ((unit_1 < 12) ? unit_1 % 4 : (unit_1 - 12) % 4);
                    kset_1 = ((unit_1 < 12) ? 4 + unit_1 / 4 : (unit_1 - 12) / 4);
                    q_row_1 = q_block_1 * 32 + lane;
                    if (head_base_1 + q_block_1 * 32 < num_heads) {
                        q_row_addr_1 = smem_qstage_addr + (unsigned int)(kset_1 * 16384) + (unsigned int)(q_row_1 * 128);
                        unsigned int words_1[8];
                        unsigned int sf_word_3 = 0;
                        unsigned int qa_1[4];
                        unsigned int qb_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 3]))
                            : "r"(q_row_addr_1 + (0 ^ q_row_1 % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 3]))
                            : "r"(q_row_addr_1 + (1 ^ q_row_1 % 8) * 16));
                        float qv_1[16];
                        qv_1[0] = __uint_as_float(qa_1[0] << 16);
                        qv_1[1] = __uint_as_float(qa_1[0] & 4294901760u);
                        qv_1[8] = __uint_as_float(qb_2[0] << 16);
                        qv_1[9] = __uint_as_float(qb_2[0] & 4294901760u);
                        qv_1[2] = __uint_as_float(qa_1[1] << 16);
                        qv_1[3] = __uint_as_float(qa_1[1] & 4294901760u);
                        qv_1[10] = __uint_as_float(qb_2[1] << 16);
                        qv_1[11] = __uint_as_float(qb_2[1] & 4294901760u);
                        qv_1[4] = __uint_as_float(qa_1[2] << 16);
                        qv_1[5] = __uint_as_float(qa_1[2] & 4294901760u);
                        qv_1[12] = __uint_as_float(qb_2[2] << 16);
                        qv_1[13] = __uint_as_float(qb_2[2] & 4294901760u);
                        qv_1[6] = __uint_as_float(qa_1[3] << 16);
                        qv_1[7] = __uint_as_float(qa_1[3] & 4294901760u);
                        qv_1[14] = __uint_as_float(qb_2[3] << 16);
                        qv_1[15] = __uint_as_float(qb_2[3] & 4294901760u);
                        float m8_1[8];
                        float _fabs_64 = fabsf(qv_1[0]);
                        float _fabs_65 = fabsf(qv_1[1]);
                        float _max_67 = max_noftz(_fabs_64, _fabs_65);
                        m8_1[0] = _max_67;
                        float _fabs_66 = fabsf(qv_1[2]);
                        float _fabs_67 = fabsf(qv_1[3]);
                        float _max_68 = max_noftz(_fabs_66, _fabs_67);
                        m8_1[1] = _max_68;
                        float _fabs_68 = fabsf(qv_1[4]);
                        float _fabs_69 = fabsf(qv_1[5]);
                        float _max_69 = max_noftz(_fabs_68, _fabs_69);
                        m8_1[2] = _max_69;
                        float _fabs_70 = fabsf(qv_1[6]);
                        float _fabs_71 = fabsf(qv_1[7]);
                        float _max_70 = max_noftz(_fabs_70, _fabs_71);
                        m8_1[3] = _max_70;
                        float _fabs_72 = fabsf(qv_1[8]);
                        float _fabs_73 = fabsf(qv_1[9]);
                        float _max_71 = max_noftz(_fabs_72, _fabs_73);
                        m8_1[4] = _max_71;
                        float _fabs_74 = fabsf(qv_1[10]);
                        float _fabs_75 = fabsf(qv_1[11]);
                        float _max_72 = max_noftz(_fabs_74, _fabs_75);
                        m8_1[5] = _max_72;
                        float _fabs_76 = fabsf(qv_1[12]);
                        float _fabs_77 = fabsf(qv_1[13]);
                        float _max_73 = max_noftz(_fabs_76, _fabs_77);
                        m8_1[6] = _max_73;
                        float _fabs_78 = fabsf(qv_1[14]);
                        float _fabs_79 = fabsf(qv_1[15]);
                        float _max_74 = max_noftz(_fabs_78, _fabs_79);
                        m8_1[7] = _max_74;
                        float m4_1[4];
                        float _max_75 = max_noftz(m8_1[0], m8_1[1]);
                        m4_1[0] = _max_75;
                        float _max_76 = max_noftz(m8_1[2], m8_1[3]);
                        m4_1[1] = _max_76;
                        float _max_77 = max_noftz(m8_1[4], m8_1[5]);
                        m4_1[2] = _max_77;
                        float _max_78 = max_noftz(m8_1[6], m8_1[7]);
                        m4_1[3] = _max_78;
                        float _max_79 = max_noftz(m4_1[0], m4_1[1]);
                        float _max_80 = max_noftz(m4_1[2], m4_1[3]);
                        float _max_81 = max_noftz(_max_79, _max_80);
                        float amax_1 = _max_81;
                        float sc_1 = amax_1 * inv_six_1;
                        uint16_t _e4m3x2_f32_20;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_20) : "f"(0.0f), "f"(sc_1));
                        uint16_t sc_pair_1 = _e4m3x2_f32_20;
                        unsigned int sc_byte_1 = (unsigned int)sc_pair_1 & 255;
                        unsigned int sc_exp_1 = sc_byte_1 >> 3 & 15;
                        unsigned int sc_man_1 = sc_byte_1 & 7;
                        float sc_norm_1 = __uint_as_float(sc_exp_1 + 120 << 23 | sc_man_1 << 20);
                        float sc_sub_1 = (float)sc_man_1 * 0.001953125f;
                        float sc_dec_1 = ((sc_exp_1 == 0) ? sc_sub_1 : sc_norm_1);
                        float _rcp_5 = __frcp_rn(sc_dec_1);
                        float inv_1 = ((sc_dec_1 > 0.0f) ? _rcp_5 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_0 = {inv_1, inv_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_1)[_ls], _scale2_0);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv_1[_ls] = qv_1[_ls] * inv_1;
                        }
                        #endif
                        uint32_t _fp4_pair_32;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_32) : "f"(qv_1[0]), "f"(qv_1[1]));
                        uint32_t _fp4_pair_33;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_33) : "f"(qv_1[2]), "f"(qv_1[3]));
                        uint32_t _fp4_pair_34;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_34) : "f"(qv_1[4]), "f"(qv_1[5]));
                        uint32_t _fp4_pair_35;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_35) : "f"(qv_1[6]), "f"(qv_1[7]));
                        uint32_t _fp4_pair_36;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_36) : "f"(qv_1[8]), "f"(qv_1[9]));
                        uint32_t _fp4_pair_37;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_37) : "f"(qv_1[10]), "f"(qv_1[11]));
                        uint32_t _fp4_pair_38;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_38) : "f"(qv_1[12]), "f"(qv_1[13]));
                        uint32_t _fp4_pair_39;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_39) : "f"(qv_1[14]), "f"(qv_1[15]));
                        words_1[0] = _fp4_pair_32 | _fp4_pair_33 << 8 | _fp4_pair_34 << 16 | _fp4_pair_35 << 24;
                        words_1[1] = _fp4_pair_36 | _fp4_pair_37 << 8 | _fp4_pair_38 << 16 | _fp4_pair_39 << 24;
                        sf_word_3 = sf_word_3 | sc_byte_1;
                        unsigned int qa_0_1[4];
                        unsigned int qb_1_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 3]))
                            : "r"(q_row_addr_1 + (2 ^ q_row_1 % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 3]))
                            : "r"(q_row_addr_1 + (3 ^ q_row_1 % 8) * 16));
                        float qv_2_1[16];
                        qv_2_1[0] = __uint_as_float(qa_0_1[0] << 16);
                        qv_2_1[1] = __uint_as_float(qa_0_1[0] & 4294901760u);
                        qv_2_1[8] = __uint_as_float(qb_1_1[0] << 16);
                        qv_2_1[9] = __uint_as_float(qb_1_1[0] & 4294901760u);
                        qv_2_1[2] = __uint_as_float(qa_0_1[1] << 16);
                        qv_2_1[3] = __uint_as_float(qa_0_1[1] & 4294901760u);
                        qv_2_1[10] = __uint_as_float(qb_1_1[1] << 16);
                        qv_2_1[11] = __uint_as_float(qb_1_1[1] & 4294901760u);
                        qv_2_1[4] = __uint_as_float(qa_0_1[2] << 16);
                        qv_2_1[5] = __uint_as_float(qa_0_1[2] & 4294901760u);
                        qv_2_1[12] = __uint_as_float(qb_1_1[2] << 16);
                        qv_2_1[13] = __uint_as_float(qb_1_1[2] & 4294901760u);
                        qv_2_1[6] = __uint_as_float(qa_0_1[3] << 16);
                        qv_2_1[7] = __uint_as_float(qa_0_1[3] & 4294901760u);
                        qv_2_1[14] = __uint_as_float(qb_1_1[3] << 16);
                        qv_2_1[15] = __uint_as_float(qb_1_1[3] & 4294901760u);
                        float m8_3_1[8];
                        float _fabs_80 = fabsf(qv_2_1[0]);
                        float _fabs_81 = fabsf(qv_2_1[1]);
                        float _max_82 = max_noftz(_fabs_80, _fabs_81);
                        m8_3_1[0] = _max_82;
                        float _fabs_82 = fabsf(qv_2_1[2]);
                        float _fabs_83 = fabsf(qv_2_1[3]);
                        float _max_83 = max_noftz(_fabs_82, _fabs_83);
                        m8_3_1[1] = _max_83;
                        float _fabs_84 = fabsf(qv_2_1[4]);
                        float _fabs_85 = fabsf(qv_2_1[5]);
                        float _max_84 = max_noftz(_fabs_84, _fabs_85);
                        m8_3_1[2] = _max_84;
                        float _fabs_86 = fabsf(qv_2_1[6]);
                        float _fabs_87 = fabsf(qv_2_1[7]);
                        float _max_85 = max_noftz(_fabs_86, _fabs_87);
                        m8_3_1[3] = _max_85;
                        float _fabs_88 = fabsf(qv_2_1[8]);
                        float _fabs_89 = fabsf(qv_2_1[9]);
                        float _max_86 = max_noftz(_fabs_88, _fabs_89);
                        m8_3_1[4] = _max_86;
                        float _fabs_90 = fabsf(qv_2_1[10]);
                        float _fabs_91 = fabsf(qv_2_1[11]);
                        float _max_87 = max_noftz(_fabs_90, _fabs_91);
                        m8_3_1[5] = _max_87;
                        float _fabs_92 = fabsf(qv_2_1[12]);
                        float _fabs_93 = fabsf(qv_2_1[13]);
                        float _max_88 = max_noftz(_fabs_92, _fabs_93);
                        m8_3_1[6] = _max_88;
                        float _fabs_94 = fabsf(qv_2_1[14]);
                        float _fabs_95 = fabsf(qv_2_1[15]);
                        float _max_89 = max_noftz(_fabs_94, _fabs_95);
                        m8_3_1[7] = _max_89;
                        float m4_4_1[4];
                        float _max_90 = max_noftz(m8_3_1[0], m8_3_1[1]);
                        m4_4_1[0] = _max_90;
                        float _max_91 = max_noftz(m8_3_1[2], m8_3_1[3]);
                        m4_4_1[1] = _max_91;
                        float _max_92 = max_noftz(m8_3_1[4], m8_3_1[5]);
                        m4_4_1[2] = _max_92;
                        float _max_93 = max_noftz(m8_3_1[6], m8_3_1[7]);
                        m4_4_1[3] = _max_93;
                        float _max_94 = max_noftz(m4_4_1[0], m4_4_1[1]);
                        float _max_95 = max_noftz(m4_4_1[2], m4_4_1[3]);
                        float _max_96 = max_noftz(_max_94, _max_95);
                        float amax_5_1 = _max_96;
                        float sc_6_1 = amax_5_1 * inv_six_1;
                        uint16_t _e4m3x2_f32_21;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_21) : "f"(0.0f), "f"(sc_6_1));
                        uint16_t sc_pair_7_1 = _e4m3x2_f32_21;
                        unsigned int sc_byte_8_1 = (unsigned int)sc_pair_7_1 & 255;
                        unsigned int sc_exp_9_1 = sc_byte_8_1 >> 3 & 15;
                        unsigned int sc_man_10_1 = sc_byte_8_1 & 7;
                        float sc_norm_11_1 = __uint_as_float(sc_exp_9_1 + 120 << 23 | sc_man_10_1 << 20);
                        float sc_sub_12_1 = (float)sc_man_10_1 * 0.001953125f;
                        float sc_dec_13_1 = ((sc_exp_9_1 == 0) ? sc_sub_12_1 : sc_norm_11_1);
                        float _rcp_6 = __frcp_rn(sc_dec_13_1);
                        float inv_14_1 = ((sc_dec_13_1 > 0.0f) ? _rcp_6 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_1 = {inv_14_1, inv_14_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2_1)[_ls], _scale2_1);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv_2_1[_ls] = qv_2_1[_ls] * inv_14_1;
                        }
                        #endif
                        uint32_t _fp4_pair_40;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_40) : "f"(qv_2_1[0]), "f"(qv_2_1[1]));
                        uint32_t _fp4_pair_41;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_41) : "f"(qv_2_1[2]), "f"(qv_2_1[3]));
                        uint32_t _fp4_pair_42;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_42) : "f"(qv_2_1[4]), "f"(qv_2_1[5]));
                        uint32_t _fp4_pair_43;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_43) : "f"(qv_2_1[6]), "f"(qv_2_1[7]));
                        uint32_t _fp4_pair_44;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_44) : "f"(qv_2_1[8]), "f"(qv_2_1[9]));
                        uint32_t _fp4_pair_45;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_45) : "f"(qv_2_1[10]), "f"(qv_2_1[11]));
                        uint32_t _fp4_pair_46;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_46) : "f"(qv_2_1[12]), "f"(qv_2_1[13]));
                        uint32_t _fp4_pair_47;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_47) : "f"(qv_2_1[14]), "f"(qv_2_1[15]));
                        words_1[2] = _fp4_pair_40 | _fp4_pair_41 << 8 | _fp4_pair_42 << 16 | _fp4_pair_43 << 24;
                        words_1[3] = _fp4_pair_44 | _fp4_pair_45 << 8 | _fp4_pair_46 << 16 | _fp4_pair_47 << 24;
                        sf_word_3 = sf_word_3 | sc_byte_8_1 << 8;
                        unsigned int qa_15_1[4];
                        unsigned int qb_16_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa_15_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_1[(0) + 3]))
                            : "r"(q_row_addr_1 + (4 ^ q_row_1 % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb_16_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_1[(0) + 3]))
                            : "r"(q_row_addr_1 + (5 ^ q_row_1 % 8) * 16));
                        float qv_17_1[16];
                        qv_17_1[0] = __uint_as_float(qa_15_1[0] << 16);
                        qv_17_1[1] = __uint_as_float(qa_15_1[0] & 4294901760u);
                        qv_17_1[8] = __uint_as_float(qb_16_1[0] << 16);
                        qv_17_1[9] = __uint_as_float(qb_16_1[0] & 4294901760u);
                        qv_17_1[2] = __uint_as_float(qa_15_1[1] << 16);
                        qv_17_1[3] = __uint_as_float(qa_15_1[1] & 4294901760u);
                        qv_17_1[10] = __uint_as_float(qb_16_1[1] << 16);
                        qv_17_1[11] = __uint_as_float(qb_16_1[1] & 4294901760u);
                        qv_17_1[4] = __uint_as_float(qa_15_1[2] << 16);
                        qv_17_1[5] = __uint_as_float(qa_15_1[2] & 4294901760u);
                        qv_17_1[12] = __uint_as_float(qb_16_1[2] << 16);
                        qv_17_1[13] = __uint_as_float(qb_16_1[2] & 4294901760u);
                        qv_17_1[6] = __uint_as_float(qa_15_1[3] << 16);
                        qv_17_1[7] = __uint_as_float(qa_15_1[3] & 4294901760u);
                        qv_17_1[14] = __uint_as_float(qb_16_1[3] << 16);
                        qv_17_1[15] = __uint_as_float(qb_16_1[3] & 4294901760u);
                        float m8_18_1[8];
                        float _fabs_96 = fabsf(qv_17_1[0]);
                        float _fabs_97 = fabsf(qv_17_1[1]);
                        float _max_97 = max_noftz(_fabs_96, _fabs_97);
                        m8_18_1[0] = _max_97;
                        float _fabs_98 = fabsf(qv_17_1[2]);
                        float _fabs_99 = fabsf(qv_17_1[3]);
                        float _max_98 = max_noftz(_fabs_98, _fabs_99);
                        m8_18_1[1] = _max_98;
                        float _fabs_100 = fabsf(qv_17_1[4]);
                        float _fabs_101 = fabsf(qv_17_1[5]);
                        float _max_99 = max_noftz(_fabs_100, _fabs_101);
                        m8_18_1[2] = _max_99;
                        float _fabs_102 = fabsf(qv_17_1[6]);
                        float _fabs_103 = fabsf(qv_17_1[7]);
                        float _max_100 = max_noftz(_fabs_102, _fabs_103);
                        m8_18_1[3] = _max_100;
                        float _fabs_104 = fabsf(qv_17_1[8]);
                        float _fabs_105 = fabsf(qv_17_1[9]);
                        float _max_101 = max_noftz(_fabs_104, _fabs_105);
                        m8_18_1[4] = _max_101;
                        float _fabs_106 = fabsf(qv_17_1[10]);
                        float _fabs_107 = fabsf(qv_17_1[11]);
                        float _max_102 = max_noftz(_fabs_106, _fabs_107);
                        m8_18_1[5] = _max_102;
                        float _fabs_108 = fabsf(qv_17_1[12]);
                        float _fabs_109 = fabsf(qv_17_1[13]);
                        float _max_103 = max_noftz(_fabs_108, _fabs_109);
                        m8_18_1[6] = _max_103;
                        float _fabs_110 = fabsf(qv_17_1[14]);
                        float _fabs_111 = fabsf(qv_17_1[15]);
                        float _max_104 = max_noftz(_fabs_110, _fabs_111);
                        m8_18_1[7] = _max_104;
                        float m4_19_1[4];
                        float _max_105 = max_noftz(m8_18_1[0], m8_18_1[1]);
                        m4_19_1[0] = _max_105;
                        float _max_106 = max_noftz(m8_18_1[2], m8_18_1[3]);
                        m4_19_1[1] = _max_106;
                        float _max_107 = max_noftz(m8_18_1[4], m8_18_1[5]);
                        m4_19_1[2] = _max_107;
                        float _max_108 = max_noftz(m8_18_1[6], m8_18_1[7]);
                        m4_19_1[3] = _max_108;
                        float _max_109 = max_noftz(m4_19_1[0], m4_19_1[1]);
                        float _max_110 = max_noftz(m4_19_1[2], m4_19_1[3]);
                        float _max_111 = max_noftz(_max_109, _max_110);
                        float amax_20_1 = _max_111;
                        float sc_21_1 = amax_20_1 * inv_six_1;
                        uint16_t _e4m3x2_f32_22;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_22) : "f"(0.0f), "f"(sc_21_1));
                        uint16_t sc_pair_22_1 = _e4m3x2_f32_22;
                        unsigned int sc_byte_23_1 = (unsigned int)sc_pair_22_1 & 255;
                        unsigned int sc_exp_24_1 = sc_byte_23_1 >> 3 & 15;
                        unsigned int sc_man_25_1 = sc_byte_23_1 & 7;
                        float sc_norm_26_1 = __uint_as_float(sc_exp_24_1 + 120 << 23 | sc_man_25_1 << 20);
                        float sc_sub_27_1 = (float)sc_man_25_1 * 0.001953125f;
                        float sc_dec_28_1 = ((sc_exp_24_1 == 0) ? sc_sub_27_1 : sc_norm_26_1);
                        float _rcp_7 = __frcp_rn(sc_dec_28_1);
                        float inv_29_1 = ((sc_dec_28_1 > 0.0f) ? _rcp_7 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_2 = {inv_29_1, inv_29_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_17_1)[_ls], _scale2_2);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv_17_1[_ls] = qv_17_1[_ls] * inv_29_1;
                        }
                        #endif
                        uint32_t _fp4_pair_48;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_48) : "f"(qv_17_1[0]), "f"(qv_17_1[1]));
                        uint32_t _fp4_pair_49;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_49) : "f"(qv_17_1[2]), "f"(qv_17_1[3]));
                        uint32_t _fp4_pair_50;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_50) : "f"(qv_17_1[4]), "f"(qv_17_1[5]));
                        uint32_t _fp4_pair_51;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_51) : "f"(qv_17_1[6]), "f"(qv_17_1[7]));
                        uint32_t _fp4_pair_52;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_52) : "f"(qv_17_1[8]), "f"(qv_17_1[9]));
                        uint32_t _fp4_pair_53;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_53) : "f"(qv_17_1[10]), "f"(qv_17_1[11]));
                        uint32_t _fp4_pair_54;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_54) : "f"(qv_17_1[12]), "f"(qv_17_1[13]));
                        uint32_t _fp4_pair_55;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_55) : "f"(qv_17_1[14]), "f"(qv_17_1[15]));
                        words_1[4] = _fp4_pair_48 | _fp4_pair_49 << 8 | _fp4_pair_50 << 16 | _fp4_pair_51 << 24;
                        words_1[5] = _fp4_pair_52 | _fp4_pair_53 << 8 | _fp4_pair_54 << 16 | _fp4_pair_55 << 24;
                        sf_word_3 = sf_word_3 | sc_byte_23_1 << 16;
                        unsigned int qa_30_1[4];
                        unsigned int qb_31_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa_30_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_1[(0) + 3]))
                            : "r"(q_row_addr_1 + (6 ^ q_row_1 % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb_31_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_1[(0) + 3]))
                            : "r"(q_row_addr_1 + (7 ^ q_row_1 % 8) * 16));
                        float qv_32_1[16];
                        qv_32_1[0] = __uint_as_float(qa_30_1[0] << 16);
                        qv_32_1[1] = __uint_as_float(qa_30_1[0] & 4294901760u);
                        qv_32_1[8] = __uint_as_float(qb_31_1[0] << 16);
                        qv_32_1[9] = __uint_as_float(qb_31_1[0] & 4294901760u);
                        qv_32_1[2] = __uint_as_float(qa_30_1[1] << 16);
                        qv_32_1[3] = __uint_as_float(qa_30_1[1] & 4294901760u);
                        qv_32_1[10] = __uint_as_float(qb_31_1[1] << 16);
                        qv_32_1[11] = __uint_as_float(qb_31_1[1] & 4294901760u);
                        qv_32_1[4] = __uint_as_float(qa_30_1[2] << 16);
                        qv_32_1[5] = __uint_as_float(qa_30_1[2] & 4294901760u);
                        qv_32_1[12] = __uint_as_float(qb_31_1[2] << 16);
                        qv_32_1[13] = __uint_as_float(qb_31_1[2] & 4294901760u);
                        qv_32_1[6] = __uint_as_float(qa_30_1[3] << 16);
                        qv_32_1[7] = __uint_as_float(qa_30_1[3] & 4294901760u);
                        qv_32_1[14] = __uint_as_float(qb_31_1[3] << 16);
                        qv_32_1[15] = __uint_as_float(qb_31_1[3] & 4294901760u);
                        float m8_33_1[8];
                        float _fabs_112 = fabsf(qv_32_1[0]);
                        float _fabs_113 = fabsf(qv_32_1[1]);
                        float _max_112 = max_noftz(_fabs_112, _fabs_113);
                        m8_33_1[0] = _max_112;
                        float _fabs_114 = fabsf(qv_32_1[2]);
                        float _fabs_115 = fabsf(qv_32_1[3]);
                        float _max_113 = max_noftz(_fabs_114, _fabs_115);
                        m8_33_1[1] = _max_113;
                        float _fabs_116 = fabsf(qv_32_1[4]);
                        float _fabs_117 = fabsf(qv_32_1[5]);
                        float _max_114 = max_noftz(_fabs_116, _fabs_117);
                        m8_33_1[2] = _max_114;
                        float _fabs_118 = fabsf(qv_32_1[6]);
                        float _fabs_119 = fabsf(qv_32_1[7]);
                        float _max_115 = max_noftz(_fabs_118, _fabs_119);
                        m8_33_1[3] = _max_115;
                        float _fabs_120 = fabsf(qv_32_1[8]);
                        float _fabs_121 = fabsf(qv_32_1[9]);
                        float _max_116 = max_noftz(_fabs_120, _fabs_121);
                        m8_33_1[4] = _max_116;
                        float _fabs_122 = fabsf(qv_32_1[10]);
                        float _fabs_123 = fabsf(qv_32_1[11]);
                        float _max_117 = max_noftz(_fabs_122, _fabs_123);
                        m8_33_1[5] = _max_117;
                        float _fabs_124 = fabsf(qv_32_1[12]);
                        float _fabs_125 = fabsf(qv_32_1[13]);
                        float _max_118 = max_noftz(_fabs_124, _fabs_125);
                        m8_33_1[6] = _max_118;
                        float _fabs_126 = fabsf(qv_32_1[14]);
                        float _fabs_127 = fabsf(qv_32_1[15]);
                        float _max_119 = max_noftz(_fabs_126, _fabs_127);
                        m8_33_1[7] = _max_119;
                        float m4_34_1[4];
                        float _max_120 = max_noftz(m8_33_1[0], m8_33_1[1]);
                        m4_34_1[0] = _max_120;
                        float _max_121 = max_noftz(m8_33_1[2], m8_33_1[3]);
                        m4_34_1[1] = _max_121;
                        float _max_122 = max_noftz(m8_33_1[4], m8_33_1[5]);
                        m4_34_1[2] = _max_122;
                        float _max_123 = max_noftz(m8_33_1[6], m8_33_1[7]);
                        m4_34_1[3] = _max_123;
                        float _max_124 = max_noftz(m4_34_1[0], m4_34_1[1]);
                        float _max_125 = max_noftz(m4_34_1[2], m4_34_1[3]);
                        float _max_126 = max_noftz(_max_124, _max_125);
                        float amax_35_1 = _max_126;
                        float sc_36_1 = amax_35_1 * inv_six_1;
                        uint16_t _e4m3x2_f32_23;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_23) : "f"(0.0f), "f"(sc_36_1));
                        uint16_t sc_pair_37_1 = _e4m3x2_f32_23;
                        unsigned int sc_byte_38_1 = (unsigned int)sc_pair_37_1 & 255;
                        unsigned int sc_exp_39_1 = sc_byte_38_1 >> 3 & 15;
                        unsigned int sc_man_40_1 = sc_byte_38_1 & 7;
                        float sc_norm_41_1 = __uint_as_float(sc_exp_39_1 + 120 << 23 | sc_man_40_1 << 20);
                        float sc_sub_42_1 = (float)sc_man_40_1 * 0.001953125f;
                        float sc_dec_43_1 = ((sc_exp_39_1 == 0) ? sc_sub_42_1 : sc_norm_41_1);
                        float _rcp_8 = __frcp_rn(sc_dec_43_1);
                        float inv_44_1 = ((sc_dec_43_1 > 0.0f) ? _rcp_8 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_3 = {inv_44_1, inv_44_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_32_1)[_ls], _scale2_3);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv_32_1[_ls] = qv_32_1[_ls] * inv_44_1;
                        }
                        #endif
                        uint32_t _fp4_pair_56;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_56) : "f"(qv_32_1[0]), "f"(qv_32_1[1]));
                        uint32_t _fp4_pair_57;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_57) : "f"(qv_32_1[2]), "f"(qv_32_1[3]));
                        uint32_t _fp4_pair_58;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_58) : "f"(qv_32_1[4]), "f"(qv_32_1[5]));
                        uint32_t _fp4_pair_59;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_59) : "f"(qv_32_1[6]), "f"(qv_32_1[7]));
                        uint32_t _fp4_pair_60;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_60) : "f"(qv_32_1[8]), "f"(qv_32_1[9]));
                        uint32_t _fp4_pair_61;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_61) : "f"(qv_32_1[10]), "f"(qv_32_1[11]));
                        uint32_t _fp4_pair_62;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_62) : "f"(qv_32_1[12]), "f"(qv_32_1[13]));
                        uint32_t _fp4_pair_63;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_63) : "f"(qv_32_1[14]), "f"(qv_32_1[15]));
                        words_1[6] = _fp4_pair_56 | _fp4_pair_57 << 8 | _fp4_pair_58 << 16 | _fp4_pair_59 << 24;
                        words_1[7] = _fp4_pair_60 | _fp4_pair_61 << 8 | _fp4_pair_62 << 16 | _fp4_pair_63 << 24;
                        sf_word_3 = sf_word_3 | sc_byte_38_1 << 24;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_qf4_addr + (unsigned int)(2 * kset_1 / 8 * 16384 + (q_row_1 * 128 + (2 * kset_1 % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_qf4_addr + (unsigned int)((2 * kset_1 + 1) / 8 * 16384 + (q_row_1 * 128 + ((2 * kset_1 + 1) % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words_1[4])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(4) + 3])));
                        smem_qsf32[kset_1 / 4 * 2048 + q_row_1 % 32 / 8 * 512 + kset_1 % 4 * 128 + q_row_1 % 8 * 16 + q_row_1 / 32 % 4 * 4 >> 2] = sf_word_3;
                    } else {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(2 * kset_1 / 8 * 16384 + (q_row_1 * 128 + (2 * kset_1 % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)((2 * kset_1 + 1) / 8 * 16384 + (q_row_1 * 128 + ((2 * kset_1 + 1) % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        smem_qsf32[kset_1 / 4 * 2048 + q_row_1 % 32 / 8 * 512 + kset_1 % 4 * 128 + q_row_1 % 8 * 16 + q_row_1 / 32 % 4 * 4 >> 2] = 0;
                    }
                }
                if (i_1 == 0) {
                    mbarrier_arrive(q_ready_k_addr);
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            unsigned int _phase_q_ready_0_1 = 0;
            mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0_1, 10000000);
            _phase_q_ready_0_1 ^= 1;
            float m_run_1 = -CAKE_INF;
            float l_run_1 = 0.0f;
            float psum_run_1 = 0.0f;
            unsigned int mask_words_1[4];
            int last_buf_1 = 0;
            int last_par_1 = 0;
            float sv0_1[16];
            float sv1_1[16];
            float sv2_1[16];
            unsigned int packed_p_1[4];
            unsigned int chunk_bits_1 = 0;
            float ov_1[16];
            int o_addr_1 = 0;
            float psum_v_1[1];
            mbarrier_wait_hint(tile_meta_addr, 0, 10000000);
            unsigned int my_mask_1 = smem_mask[local_warp_1];
            int valid_1 = (int)(my_mask_1 >> (unsigned int)lane & 1);
            mbarrier_wait_cluster_hint(kv_landed_addr, 0, 10000000);
            int kbase_1 = smem_v5_addr;
            int kzone_off_1 = 0;
            int vbase_1 = smem_v_addr;
            {
                if (valid_1 != 0) {
                    unsigned int sfw[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 3]))
                        : "r"(kbase_1 + (16384 + (row_1 * 128 + (96 ^ row_1 % 8 * 16)))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfw[4])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(4) + 3]))
                        : "r"(kbase_1 + (16384 + (row_1 * 128 + (112 ^ row_1 % 8 * 16)))));
                    smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw[0];
                    smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw[1];
                    smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw[2];
                    smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw[3];
                    smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw[4];
                    smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw[5];
                    smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw[6];
                    smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                } else {
                    smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                    smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                    smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                    smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                    smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                    smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                    smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                    smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(k_ready_addr);
            }
            unsigned int zero4_1[4];
            zero4_1[0] = 0;
            zero4_1[1] = 0;
            zero4_1[2] = 0;
            zero4_1[3] = 0;
            int znope_hi_v_2 = 14;
            int zrope_pairs_hi_v_1 = 1;
            int nope_lo_1 = ((rank_1 == 0) ? 0 : 8);
            int nope_hi_v_1 = ((rank_1 == 0) ? 3 : 12);
            int rope_pair_lo_1 = 0;
            int rope_pairs_hi_v_1 = 0;
            {
                nope_lo_1 = ((rank_1 == 0) ? 3 : 12);
                nope_hi_v_1 = ((rank_1 == 0) ? 6 : 14);
                rope_pairs_hi_v_1 = ((rank_1 == 0) ? 0 : 2);
            }
            int vrank8_1 = 8 * rank_1;
            if (valid_1 != 0) {
                #pragma unroll 1
                for (int c_6 = nope_lo_1; c_6 < nope_hi_v_1; c_6++) {
                    unsigned int raw_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(0) + 3]))
                        : "r"(kbase_1 + (c_6 / 8 * 16384 + (row_1 * 128 + (c_6 % 8 * 16 ^ row_1 % 8 * 16)))));
                    int sf_w_2 = c_6 / 2;
                    unsigned int sf_word_4 = smem_kzone32[kzone_off_1 + ((14 + (sf_w_2 >> 2)) / 8 * 16384 + (row_1 * 128 + ((14 + (sf_w_2 >> 2)) % 8 * 16 ^ row_1 % 8 * 16))) + 4 * (sf_w_2 & 3) >> 2];
                    int block_2 = 2 * c_6;
                    unsigned int scale_3 = sf_word_4 >> (unsigned int)(8 * ((c_6 & 1) * 2)) & 255;
                    unsigned int v8_3[4];
                    {
                        v8_3[0] = cake_dsv4_qmul4_portable<5>(raw_2[0], scale_3);
                    }
                    {
                        v8_3[1] = cake_dsv4_qmul4_portable<6>(raw_2[0], scale_3);
                    }
                    {
                        v8_3[2] = cake_dsv4_qmul4_portable<5>(raw_2[1], scale_3);
                    }
                    {
                        v8_3[3] = cake_dsv4_qmul4_portable<6>(raw_2[1], scale_3);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_1 + (c_6 - vrank8_1) / 4 * 16384 + (row_1 * 128 + (2 * c_6 % 8 * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 3])));
                    int block_0_2 = 2 * c_6 + 1;
                    unsigned int scale_1_2 = sf_word_4 >> (unsigned int)(8 * ((c_6 & 1) * 2 + 1)) & 255;
                    unsigned int v8_2_2[4];
                    {
                        v8_2_2[0] = cake_dsv4_qmul4_portable<5>(raw_2[2], scale_1_2);
                    }
                    {
                        v8_2_2[1] = cake_dsv4_qmul4_portable<6>(raw_2[2], scale_1_2);
                    }
                    {
                        v8_2_2[2] = cake_dsv4_qmul4_portable<5>(raw_2[3], scale_1_2);
                    }
                    {
                        v8_2_2[3] = cake_dsv4_qmul4_portable<6>(raw_2[3], scale_1_2);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_1 + (c_6 - vrank8_1) / 4 * 16384 + (row_1 * 128 + ((2 * c_6 + 1) % 8 * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_2_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_2[(0) + 3])));
                }
                for (int jp_4 = rope_pair_lo_1; jp_4 < rope_pairs_hi_v_1; jp_4++) {
                    unsigned int vrope_2[4];
                    int j_2 = 2 * jp_4;
                    unsigned int rope_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 3]))
                        : "r"(kbase_1 + 36864 + (row_1 * 128 + (j_2 * 16 ^ row_1 % 8 * 16))));
                    float lo_2 = __uint_as_float(rope_2[0] << 16);
                    float hi_3 = __uint_as_float(rope_2[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_24;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_24) : "f"(hi_3), "f"(lo_2));
                    uint16_t pair_3 = _e4m3x2_f32_24;
                    {
                        vrope_2[0] = (unsigned int)pair_3;
                    }
                    float lo_0_2 = __uint_as_float(rope_2[1] << 16);
                    float hi_1_2 = __uint_as_float(rope_2[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_25;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_25) : "f"(hi_1_2), "f"(lo_0_2));
                    uint16_t pair_2_2 = _e4m3x2_f32_25;
                    {
                        vrope_2[0] = vrope_2[0] | (unsigned int)pair_2_2 << 16;
                    }
                    float lo_3_2 = __uint_as_float(rope_2[2] << 16);
                    float hi_4_2 = __uint_as_float(rope_2[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_26;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_26) : "f"(hi_4_2), "f"(lo_3_2));
                    uint16_t pair_5_2 = _e4m3x2_f32_26;
                    {
                        vrope_2[1] = (unsigned int)pair_5_2;
                    }
                    float lo_6_2 = __uint_as_float(rope_2[3] << 16);
                    float hi_7_2 = __uint_as_float(rope_2[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_27;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_27) : "f"(hi_7_2), "f"(lo_6_2));
                    uint16_t pair_8_2 = _e4m3x2_f32_27;
                    {
                        vrope_2[1] = vrope_2[1] | (unsigned int)pair_8_2 << 16;
                    }
                    int j_9_2 = 2 * jp_4 + 1;
                    unsigned int rope_10_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_10_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_2[(0) + 3]))
                        : "r"(kbase_1 + 36864 + (row_1 * 128 + (j_9_2 * 16 ^ row_1 % 8 * 16))));
                    float lo_11_2 = __uint_as_float(rope_10_2[0] << 16);
                    float hi_12_2 = __uint_as_float(rope_10_2[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_28;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_28) : "f"(hi_12_2), "f"(lo_11_2));
                    uint16_t pair_13_2 = _e4m3x2_f32_28;
                    {
                        vrope_2[2] = (unsigned int)pair_13_2;
                    }
                    float lo_14_2 = __uint_as_float(rope_10_2[1] << 16);
                    float hi_15_2 = __uint_as_float(rope_10_2[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_29;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_29) : "f"(hi_15_2), "f"(lo_14_2));
                    uint16_t pair_16_2 = _e4m3x2_f32_29;
                    {
                        vrope_2[2] = vrope_2[2] | (unsigned int)pair_16_2 << 16;
                    }
                    float lo_17_2 = __uint_as_float(rope_10_2[2] << 16);
                    float hi_18_2 = __uint_as_float(rope_10_2[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_30;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_30) : "f"(hi_18_2), "f"(lo_17_2));
                    uint16_t pair_19_2 = _e4m3x2_f32_30;
                    {
                        vrope_2[3] = (unsigned int)pair_19_2;
                    }
                    float lo_20_2 = __uint_as_float(rope_10_2[3] << 16);
                    float hi_21_2 = __uint_as_float(rope_10_2[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_31;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_31) : "f"(hi_21_2), "f"(lo_20_2));
                    uint16_t pair_22_2 = _e4m3x2_f32_31;
                    {
                        vrope_2[3] = vrope_2[3] | (unsigned int)pair_22_2 << 16;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_1 + 16384 + (row_1 * 128 + ((4 + jp_4) * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&vrope_2[0])), "r"(*reinterpret_cast<uint32_t*>(&vrope_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vrope_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vrope_2[(0) + 3])));
                }
            } else {
                {
                    for (int c_7 = 8; c_7 < znope_hi_v_2; c_7++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(kbase_1 + (c_7 / 8 * 16384 + (row_1 * 128 + (c_7 % 8 * 16 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 3])));
                    }
                }
                {
                    for (int jp_5 = 0; jp_5 < zrope_pairs_hi_v_1; jp_5++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(kbase_1 + 36864 + (row_1 * 128 + (2 * jp_5 * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(kbase_1 + 36864 + (row_1 * 128 + ((2 * jp_5 + 1) * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 3])));
                    }
                }
                for (int c_8 = nope_lo_1; c_8 < nope_hi_v_1; c_8++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_1 + (c_8 - vrank8_1) / 4 * 16384 + (row_1 * 128 + (2 * c_8 % 8 * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_1 + (c_8 - vrank8_1) / 4 * 16384 + (row_1 * 128 + ((2 * c_8 + 1) % 8 * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 3])));
                }
                for (int jp_6 = rope_pair_lo_1; jp_6 < rope_pairs_hi_v_1; jp_6++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_1 + 16384 + (row_1 * 128 + ((4 + jp_6) * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_1[(0) + 3])));
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            #pragma unroll 1
            for (int t_1 = tile_lo_1; t_1 < tile_hi_1; t_1++) {
                int it_1 = t_1 - tile_lo_1;
                int buf_1 = it_1 & 1;
                int par_1 = it_1 >> 1 & 1;
                if (it_1 > 0) {
                    mbarrier_wait_hint(pv_done_addr + (last_buf_1) * 8, last_par_1, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(kbuf_free_p_addr + (last_buf_1) * 8);
                }
                mask_words_1[0] = smem_mask[(it_1 & 3) * 4];
                mask_words_1[1] = smem_mask[(it_1 & 3) * 4 + 1];
                mask_words_1[2] = smem_mask[(it_1 & 3) * 4 + 2];
                mask_words_1[3] = smem_mask[(it_1 & 3) * 4 + 3];
                kbase_1 = smem_v5_addr + (unsigned int)(buf_1 * 118784);
                mbarrier_wait_hint(s_full_addr + (buf_1) * 8, par_1, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp_rows_valid_1 != 0) {
                    tmem_ld_x16(&sv0_1[0], taddr + 256 + 48 + (unsigned int)(tmem_row_origin_1 << 16));
                    tmem_ld_x16(&sv1_1[0], taddr + 256 + 48 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                    tmem_ld_x16(&sv2_1[0], taddr + 256 + 48 + 32 + (unsigned int)(tmem_row_origin_1 << 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                }
                mbarrier_arrive(s_free_addr);
                mbarrier_arrive(kbuf_free_k_addr + (buf_1) * 8);
                float slice_max_1 = -CAKE_INF;
                if (warp_rows_valid_1 != 0) {
                    chunk_bits_1 = mask_words_1[1] >> 16 & 65535;
                    if (chunk_bits_1 != 65535) {
                        if ((chunk_bits_1 & 1) == 0) {
                            sv0_1[0] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 1 & 1) == 0) {
                            sv0_1[1] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 2 & 1) == 0) {
                            sv0_1[2] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 3 & 1) == 0) {
                            sv0_1[3] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 4 & 1) == 0) {
                            sv0_1[4] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 5 & 1) == 0) {
                            sv0_1[5] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 6 & 1) == 0) {
                            sv0_1[6] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 7 & 1) == 0) {
                            sv0_1[7] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 8 & 1) == 0) {
                            sv0_1[8] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 9 & 1) == 0) {
                            sv0_1[9] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 10 & 1) == 0) {
                            sv0_1[10] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 11 & 1) == 0) {
                            sv0_1[11] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 12 & 1) == 0) {
                            sv0_1[12] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 13 & 1) == 0) {
                            sv0_1[13] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 14 & 1) == 0) {
                            sv0_1[14] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 15 & 1) == 0) {
                            sv0_1[15] = -CAKE_INF;
                        }
                    }
                    float sv0_max_1 = sv0_1[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv0_max_1 = max_noftz(sv0_max_1, sv0_1[_lr]);
                    }
                    float _max_127 = max_noftz(slice_max_1, sv0_max_1);
                    slice_max_1 = _max_127;
                    chunk_bits_1 = mask_words_1[2] & 65535;
                    if (chunk_bits_1 != 65535) {
                        if ((chunk_bits_1 & 1) == 0) {
                            sv1_1[0] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 1 & 1) == 0) {
                            sv1_1[1] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 2 & 1) == 0) {
                            sv1_1[2] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 3 & 1) == 0) {
                            sv1_1[3] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 4 & 1) == 0) {
                            sv1_1[4] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 5 & 1) == 0) {
                            sv1_1[5] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 6 & 1) == 0) {
                            sv1_1[6] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 7 & 1) == 0) {
                            sv1_1[7] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 8 & 1) == 0) {
                            sv1_1[8] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 9 & 1) == 0) {
                            sv1_1[9] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 10 & 1) == 0) {
                            sv1_1[10] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 11 & 1) == 0) {
                            sv1_1[11] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 12 & 1) == 0) {
                            sv1_1[12] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 13 & 1) == 0) {
                            sv1_1[13] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 14 & 1) == 0) {
                            sv1_1[14] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 15 & 1) == 0) {
                            sv1_1[15] = -CAKE_INF;
                        }
                    }
                    float sv1_max_1 = sv1_1[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv1_max_1 = max_noftz(sv1_max_1, sv1_1[_lr]);
                    }
                    float _max_128 = max_noftz(slice_max_1, sv1_max_1);
                    slice_max_1 = _max_128;
                    chunk_bits_1 = mask_words_1[2] >> 16 & 65535;
                    if (chunk_bits_1 != 65535) {
                        if ((chunk_bits_1 & 1) == 0) {
                            sv2_1[0] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 1 & 1) == 0) {
                            sv2_1[1] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 2 & 1) == 0) {
                            sv2_1[2] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 3 & 1) == 0) {
                            sv2_1[3] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 4 & 1) == 0) {
                            sv2_1[4] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 5 & 1) == 0) {
                            sv2_1[5] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 6 & 1) == 0) {
                            sv2_1[6] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 7 & 1) == 0) {
                            sv2_1[7] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 8 & 1) == 0) {
                            sv2_1[8] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 9 & 1) == 0) {
                            sv2_1[9] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 10 & 1) == 0) {
                            sv2_1[10] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 11 & 1) == 0) {
                            sv2_1[11] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 12 & 1) == 0) {
                            sv2_1[12] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 13 & 1) == 0) {
                            sv2_1[13] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 14 & 1) == 0) {
                            sv2_1[14] = -CAKE_INF;
                        }
                        if ((chunk_bits_1 >> 15 & 1) == 0) {
                            sv2_1[15] = -CAKE_INF;
                        }
                    }
                    float sv2_max_1 = sv2_1[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv2_max_1 = max_noftz(sv2_max_1, sv2_1[_lr]);
                    }
                    float _max_129 = max_noftz(slice_max_1, sv2_max_1);
                    slice_max_1 = _max_129;
                }
                smem_pmax[128 + row_1] = slice_max_1;
                asm volatile("barrier.sync 8, 384;" ::: "memory");
                float _max_130 = max_noftz(smem_pmax[row_1], smem_pmax[128 + row_1]);
                float _max_131 = max_noftz(_max_130, smem_pmax[256 + row_1]);
                float tile_max_1 = _max_131;
                float cand_1 = tile_max_1 * softmax_scale_log2_1;
                if (it_1 == 0) {
                    if (has_sink_row_1 != 0) {
                        float _max_132 = max_noftz(cand_1, sink_log2_1);
                        cand_1 = _max_132;
                    }
                }
                float _max_133 = max_noftz(cand_1, m_run_1);
                cand_1 = _max_133;
                int grow_1 = 0;
                if (it_1 == 0) {
                    grow_1 = 1;
                }
                if (cand_1 - m_run_1 > 8.0f) {
                    grow_1 = 1;
                }
                if (grow_1 != 0) {
                    float _exp2_2 = approx_exp2(m_run_1 - cand_1);
                    float alpha_1 = ((m_run_1 > -CAKE_INF) ? _exp2_2 : 0.0f);
                    l_run_1 = l_run_1 * alpha_1;
                    psum_run_1 = psum_run_1 * alpha_1;
                    if (it_1 > 0) {
                        if (warp_rows_valid_1 != 0) {
                            o_addr_1 = taddr + 96 + (unsigned int)(tmem_row_origin_1 << 16);
                            tmem_ld_x16(&ov_1[0], o_addr_1);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_12 = {alpha_1, alpha_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_1)[_ls], _scale2_12);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov_1[_ls] = ov_1[_ls] * alpha_1;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr_1, ov_1);
                            o_addr_1 = taddr + 96 + 16 + (unsigned int)(tmem_row_origin_1 << 16);
                            tmem_ld_x16(&ov_1[0], o_addr_1);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_13 = {alpha_1, alpha_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_1)[_ls], _scale2_13);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov_1[_ls] = ov_1[_ls] * alpha_1;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr_1, ov_1);
                            o_addr_1 = taddr + 96 + 32 + (unsigned int)(tmem_row_origin_1 << 16);
                            tmem_ld_x16(&ov_1[0], o_addr_1);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_14 = {alpha_1, alpha_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_1)[_ls], _scale2_14);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov_1[_ls] = ov_1[_ls] * alpha_1;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr_1, ov_1);
                            o_addr_1 = taddr + 96 + 48 + (unsigned int)(tmem_row_origin_1 << 16);
                            tmem_ld_x16(&ov_1[0], o_addr_1);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_15 = {alpha_1, alpha_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_1)[_ls], _scale2_15);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov_1[_ls] = ov_1[_ls] * alpha_1;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr_1, ov_1);
                            o_addr_1 = taddr + 96 + 64 + (unsigned int)(tmem_row_origin_1 << 16);
                            tmem_ld_x16(&ov_1[0], o_addr_1);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_16 = {alpha_1, alpha_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_1)[_ls], _scale2_16);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov_1[_ls] = ov_1[_ls] * alpha_1;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr_1, ov_1);
                            o_addr_1 = taddr + 96 + 80 + (unsigned int)(tmem_row_origin_1 << 16);
                            tmem_ld_x16(&ov_1[0], o_addr_1);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_17 = {alpha_1, alpha_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_1)[_ls], _scale2_17);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov_1[_ls] = ov_1[_ls] * alpha_1;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr_1, ov_1);
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                    }
                    m_run_1 = cand_1;
                }
                float m_use_1 = ((m_run_1 > -CAKE_INF) ? m_run_1 : 0.0f);
                float slice_sum_1 = 0.0f;
                if (warp_rows_valid_1 != 0) {
                    float score_bias_1 = -m_use_1;
                    const float2 _fma_b2_18 = {softmax_scale_log2_1, softmax_scale_log2_1};
                    const float2 _fma_c2_19 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv0_1)[_lf], _fma_b2_18, _fma_c2_19);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv0_1[_le] = approx_exp2(sv0_1[_le]);
                    }
                    float sv0_sum_1 = 0.0f;
                    #pragma unroll
                    for (int _lr = 0; _lr < 16; _lr++) {
                        sv0_sum_1 += sv0_1[_lr];
                    }
                    slice_sum_1 = slice_sum_1 + sv0_sum_1;
                    if (row_valid_1 == 0) {
                        sv0_1[0] = 0.0f;
                        sv0_1[1] = 0.0f;
                        sv0_1[2] = 0.0f;
                        sv0_1[3] = 0.0f;
                        sv0_1[4] = 0.0f;
                        sv0_1[5] = 0.0f;
                        sv0_1[6] = 0.0f;
                        sv0_1[7] = 0.0f;
                        sv0_1[8] = 0.0f;
                        sv0_1[9] = 0.0f;
                        sv0_1[10] = 0.0f;
                        sv0_1[11] = 0.0f;
                        sv0_1[12] = 0.0f;
                        sv0_1[13] = 0.0f;
                        sv0_1[14] = 0.0f;
                        sv0_1[15] = 0.0f;
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
                            : "=r"(_packed) : "f"(sv0_1[0]), "f"(sv0_1[1]),
                                               "f"(sv0_1[2]), "f"(sv0_1[3]));
                        packed_p_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv0_1[4]), "f"(sv0_1[5]),
                                               "f"(sv0_1[6]), "f"(sv0_1[7]));
                        packed_p_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv0_1[8]), "f"(sv0_1[9]),
                                               "f"(sv0_1[10]), "f"(sv0_1[11]));
                        packed_p_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv0_1[12]), "f"(sv0_1[13]),
                                               "f"(sv0_1[14]), "f"(sv0_1[15]));
                        packed_p_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(kbase_1 + 36864 + (row_1 * 128 + (48 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 3])));
                    const float2 _fma_b2_20 = {softmax_scale_log2_1, softmax_scale_log2_1};
                    const float2 _fma_c2_21 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv1_1)[_lf], _fma_b2_20, _fma_c2_21);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv1_1[_le] = approx_exp2(sv1_1[_le]);
                    }
                    float sv1_sum_1 = 0.0f;
                    #pragma unroll
                    for (int _lr = 0; _lr < 16; _lr++) {
                        sv1_sum_1 += sv1_1[_lr];
                    }
                    slice_sum_1 = slice_sum_1 + sv1_sum_1;
                    if (row_valid_1 == 0) {
                        sv1_1[0] = 0.0f;
                        sv1_1[1] = 0.0f;
                        sv1_1[2] = 0.0f;
                        sv1_1[3] = 0.0f;
                        sv1_1[4] = 0.0f;
                        sv1_1[5] = 0.0f;
                        sv1_1[6] = 0.0f;
                        sv1_1[7] = 0.0f;
                        sv1_1[8] = 0.0f;
                        sv1_1[9] = 0.0f;
                        sv1_1[10] = 0.0f;
                        sv1_1[11] = 0.0f;
                        sv1_1[12] = 0.0f;
                        sv1_1[13] = 0.0f;
                        sv1_1[14] = 0.0f;
                        sv1_1[15] = 0.0f;
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
                            : "=r"(_packed) : "f"(sv1_1[0]), "f"(sv1_1[1]),
                                               "f"(sv1_1[2]), "f"(sv1_1[3]));
                        packed_p_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv1_1[4]), "f"(sv1_1[5]),
                                               "f"(sv1_1[6]), "f"(sv1_1[7]));
                        packed_p_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv1_1[8]), "f"(sv1_1[9]),
                                               "f"(sv1_1[10]), "f"(sv1_1[11]));
                        packed_p_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv1_1[12]), "f"(sv1_1[13]),
                                               "f"(sv1_1[14]), "f"(sv1_1[15]));
                        packed_p_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(kbase_1 + 36864 + (row_1 * 128 + (64 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 3])));
                    const float2 _fma_b2_22 = {softmax_scale_log2_1, softmax_scale_log2_1};
                    const float2 _fma_c2_23 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv2_1)[_lf], _fma_b2_22, _fma_c2_23);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv2_1[_le] = approx_exp2(sv2_1[_le]);
                    }
                    float sv2_sum_1 = 0.0f;
                    #pragma unroll
                    for (int _lr = 0; _lr < 16; _lr++) {
                        sv2_sum_1 += sv2_1[_lr];
                    }
                    slice_sum_1 = slice_sum_1 + sv2_sum_1;
                    if (row_valid_1 == 0) {
                        sv2_1[0] = 0.0f;
                        sv2_1[1] = 0.0f;
                        sv2_1[2] = 0.0f;
                        sv2_1[3] = 0.0f;
                        sv2_1[4] = 0.0f;
                        sv2_1[5] = 0.0f;
                        sv2_1[6] = 0.0f;
                        sv2_1[7] = 0.0f;
                        sv2_1[8] = 0.0f;
                        sv2_1[9] = 0.0f;
                        sv2_1[10] = 0.0f;
                        sv2_1[11] = 0.0f;
                        sv2_1[12] = 0.0f;
                        sv2_1[13] = 0.0f;
                        sv2_1[14] = 0.0f;
                        sv2_1[15] = 0.0f;
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
                            : "=r"(_packed) : "f"(sv2_1[0]), "f"(sv2_1[1]),
                                               "f"(sv2_1[2]), "f"(sv2_1[3]));
                        packed_p_1[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv2_1[4]), "f"(sv2_1[5]),
                                               "f"(sv2_1[6]), "f"(sv2_1[7]));
                        packed_p_1[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv2_1[8]), "f"(sv2_1[9]),
                                               "f"(sv2_1[10]), "f"(sv2_1[11]));
                        packed_p_1[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv2_1[12]), "f"(sv2_1[13]),
                                               "f"(sv2_1[14]), "f"(sv2_1[15]));
                        packed_p_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(kbase_1 + 36864 + (row_1 * 128 + (80 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 3])));
                }
                smem_psum[128 + row_1] = slice_sum_1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr + (buf_1) * 8);
                asm volatile("barrier.sync 8, 384;" ::: "memory");
                l_run_1 = l_run_1 + smem_psum[row_1] + smem_psum[128 + row_1] + smem_psum[256 + row_1];
                mbarrier_wait_hint(psum_full_addr + (buf_1) * 8, par_1, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp_rows_valid_1 != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                        " {%0}, [%1];"
                        : "=f"(psum_v_1[0])
                        : "r"(taddr + 448 + (unsigned int)(tmem_row_origin_1 << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    psum_run_1 = psum_run_1 + psum_v_1[0];
                }
                if (tile_hi_1 > t_1 + 1) {
                    int nbuf_1 = buf_1 ^ 1;
                    int npar_1 = it_1 + 1 >> 1 & 1;
                    mbarrier_wait_hint(tile_meta_addr + (it_1 + 1 & 3) * 8, it_1 + 1 >> 2 & 1, 10000000);
                    my_mask_1 = smem_mask[(it_1 + 1 & 3) * 4 + local_warp_1];
                    valid_1 = (int)(my_mask_1 >> (unsigned int)lane & 1);
                    mbarrier_wait_cluster_hint(kv_landed_addr + (nbuf_1) * 8, npar_1, 10000000);
                    kbase_1 = smem_v5_addr + (unsigned int)(nbuf_1 * 118784);
                    kzone_off_1 = nbuf_1 * 118784;
                    vbase_1 = smem_v_addr + (unsigned int)(nbuf_1 * 32768);
                    {
                        if (valid_1 != 0) {
                            unsigned int sfw_1[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 3]))
                                : "r"(kbase_1 + (16384 + (row_1 * 128 + (96 ^ row_1 % 8 * 16)))));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(4) + 3]))
                                : "r"(kbase_1 + (16384 + (row_1 * 128 + (112 ^ row_1 % 8 * 16)))));
                            smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw_1[0];
                            smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw_1[1];
                            smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw_1[2];
                            smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw_1[3];
                            smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw_1[4];
                            smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw_1[5];
                            smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = sfw_1[6];
                            smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                        } else {
                            smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                            smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                            smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                            smem_kzone32[kzone_off_1 + 32768 + (row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                            smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                            smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                            smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                            smem_kzone32[kzone_off_1 + 32768 + (2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4) >> 2] = 0;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(k_ready_addr + (nbuf_1) * 8);
                    }
                    unsigned int zero4_0_1[4];
                    zero4_0_1[0] = 0;
                    zero4_0_1[1] = 0;
                    zero4_0_1[2] = 0;
                    zero4_0_1[3] = 0;
                    int znope_hi_v_1_1 = 14;
                    int zrope_pairs_hi_v_2_1 = 1;
                    int nope_lo_3_1 = ((rank_1 == 0) ? 0 : 8);
                    int nope_hi_v_4_1 = ((rank_1 == 0) ? 3 : 12);
                    int rope_pair_lo_5_1 = 0;
                    int rope_pairs_hi_v_6_1 = 0;
                    {
                        nope_lo_3_1 = ((rank_1 == 0) ? 3 : 12);
                        nope_hi_v_4_1 = ((rank_1 == 0) ? 6 : 14);
                        rope_pairs_hi_v_6_1 = ((rank_1 == 0) ? 0 : 2);
                    }
                    int vrank8_7_1 = 8 * rank_1;
                    if (valid_1 != 0) {
                        #pragma unroll 1
                        for (int c_9 = nope_lo_3_1; c_9 < nope_hi_v_4_1; c_9++) {
                            unsigned int raw_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_3[(0) + 3]))
                                : "r"(kbase_1 + (c_9 / 8 * 16384 + (row_1 * 128 + (c_9 % 8 * 16 ^ row_1 % 8 * 16)))));
                            int sf_w_3 = c_9 / 2;
                            unsigned int sf_word_5 = smem_kzone32[kzone_off_1 + ((14 + (sf_w_3 >> 2)) / 8 * 16384 + (row_1 * 128 + ((14 + (sf_w_3 >> 2)) % 8 * 16 ^ row_1 % 8 * 16))) + 4 * (sf_w_3 & 3) >> 2];
                            int block_3 = 2 * c_9;
                            unsigned int scale_4 = sf_word_5 >> (unsigned int)(8 * ((c_9 & 1) * 2)) & 255;
                            unsigned int v8_4[4];
                            {
                                v8_4[0] = cake_dsv4_qmul4_portable<5>(raw_3[0], scale_4);
                            }
                            {
                                v8_4[1] = cake_dsv4_qmul4_portable<6>(raw_3[0], scale_4);
                            }
                            {
                                v8_4[2] = cake_dsv4_qmul4_portable<5>(raw_3[1], scale_4);
                            }
                            {
                                v8_4[3] = cake_dsv4_qmul4_portable<6>(raw_3[1], scale_4);
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_1 + (c_9 - vrank8_7_1) / 4 * 16384 + (row_1 * 128 + (2 * c_9 % 8 * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 3])));
                            int block_0_3 = 2 * c_9 + 1;
                            unsigned int scale_1_3 = sf_word_5 >> (unsigned int)(8 * ((c_9 & 1) * 2 + 1)) & 255;
                            unsigned int v8_2_3[4];
                            {
                                v8_2_3[0] = cake_dsv4_qmul4_portable<5>(raw_3[2], scale_1_3);
                            }
                            {
                                v8_2_3[1] = cake_dsv4_qmul4_portable<6>(raw_3[2], scale_1_3);
                            }
                            {
                                v8_2_3[2] = cake_dsv4_qmul4_portable<5>(raw_3[3], scale_1_3);
                            }
                            {
                                v8_2_3[3] = cake_dsv4_qmul4_portable<6>(raw_3[3], scale_1_3);
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_1 + (c_9 - vrank8_7_1) / 4 * 16384 + (row_1 * 128 + ((2 * c_9 + 1) % 8 * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_2_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_3[(0) + 3])));
                        }
                        for (int jp_7 = rope_pair_lo_5_1; jp_7 < rope_pairs_hi_v_6_1; jp_7++) {
                            unsigned int vrope_3[4];
                            int j_3 = 2 * jp_7;
                            unsigned int rope_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 3]))
                                : "r"(kbase_1 + 36864 + (row_1 * 128 + (j_3 * 16 ^ row_1 % 8 * 16))));
                            float lo_4 = __uint_as_float(rope_3[0] << 16);
                            float hi_5 = __uint_as_float(rope_3[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_32;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_32) : "f"(hi_5), "f"(lo_4));
                            uint16_t pair_4 = _e4m3x2_f32_32;
                            {
                                vrope_3[0] = (unsigned int)pair_4;
                            }
                            float lo_0_3 = __uint_as_float(rope_3[1] << 16);
                            float hi_1_3 = __uint_as_float(rope_3[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_33;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_33) : "f"(hi_1_3), "f"(lo_0_3));
                            uint16_t pair_2_3 = _e4m3x2_f32_33;
                            {
                                vrope_3[0] = vrope_3[0] | (unsigned int)pair_2_3 << 16;
                            }
                            float lo_3_3 = __uint_as_float(rope_3[2] << 16);
                            float hi_4_3 = __uint_as_float(rope_3[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_34;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_34) : "f"(hi_4_3), "f"(lo_3_3));
                            uint16_t pair_5_3 = _e4m3x2_f32_34;
                            {
                                vrope_3[1] = (unsigned int)pair_5_3;
                            }
                            float lo_6_3 = __uint_as_float(rope_3[3] << 16);
                            float hi_7_3 = __uint_as_float(rope_3[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_35;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_35) : "f"(hi_7_3), "f"(lo_6_3));
                            uint16_t pair_8_3 = _e4m3x2_f32_35;
                            {
                                vrope_3[1] = vrope_3[1] | (unsigned int)pair_8_3 << 16;
                            }
                            int j_9_3 = 2 * jp_7 + 1;
                            unsigned int rope_10_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_10_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_3[(0) + 3]))
                                : "r"(kbase_1 + 36864 + (row_1 * 128 + (j_9_3 * 16 ^ row_1 % 8 * 16))));
                            float lo_11_3 = __uint_as_float(rope_10_3[0] << 16);
                            float hi_12_3 = __uint_as_float(rope_10_3[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_36;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_36) : "f"(hi_12_3), "f"(lo_11_3));
                            uint16_t pair_13_3 = _e4m3x2_f32_36;
                            {
                                vrope_3[2] = (unsigned int)pair_13_3;
                            }
                            float lo_14_3 = __uint_as_float(rope_10_3[1] << 16);
                            float hi_15_3 = __uint_as_float(rope_10_3[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_37;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_37) : "f"(hi_15_3), "f"(lo_14_3));
                            uint16_t pair_16_3 = _e4m3x2_f32_37;
                            {
                                vrope_3[2] = vrope_3[2] | (unsigned int)pair_16_3 << 16;
                            }
                            float lo_17_3 = __uint_as_float(rope_10_3[2] << 16);
                            float hi_18_3 = __uint_as_float(rope_10_3[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_38;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_38) : "f"(hi_18_3), "f"(lo_17_3));
                            uint16_t pair_19_3 = _e4m3x2_f32_38;
                            {
                                vrope_3[3] = (unsigned int)pair_19_3;
                            }
                            float lo_20_3 = __uint_as_float(rope_10_3[3] << 16);
                            float hi_21_3 = __uint_as_float(rope_10_3[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_39;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_39) : "f"(hi_21_3), "f"(lo_20_3));
                            uint16_t pair_22_3 = _e4m3x2_f32_39;
                            {
                                vrope_3[3] = vrope_3[3] | (unsigned int)pair_22_3 << 16;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_1 + 16384 + (row_1 * 128 + ((4 + jp_7) * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&vrope_3[0])), "r"(*reinterpret_cast<uint32_t*>(&vrope_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vrope_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vrope_3[(0) + 3])));
                        }
                    } else {
                        {
                            for (int c_10 = 8; c_10 < znope_hi_v_1_1; c_10++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                    "r"(kbase_1 + (c_10 / 8 * 16384 + (row_1 * 128 + (c_10 % 8 * 16 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 3])));
                            }
                        }
                        {
                            for (int jp_8 = 0; jp_8 < zrope_pairs_hi_v_2_1; jp_8++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                    "r"(kbase_1 + 36864 + (row_1 * 128 + (2 * jp_8 * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 3])));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                    "r"(kbase_1 + 36864 + (row_1 * 128 + ((2 * jp_8 + 1) * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 3])));
                            }
                        }
                        for (int c_11 = nope_lo_3_1; c_11 < nope_hi_v_4_1; c_11++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_1 + (c_11 - vrank8_7_1) / 4 * 16384 + (row_1 * 128 + (2 * c_11 % 8 * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 3])));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_1 + (c_11 - vrank8_7_1) / 4 * 16384 + (row_1 * 128 + ((2 * c_11 + 1) % 8 * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 3])));
                        }
                        for (int jp_9 = rope_pair_lo_5_1; jp_9 < rope_pairs_hi_v_6_1; jp_9++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_1 + 16384 + (row_1 * 128 + ((4 + jp_9) * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_1[(0) + 3])));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_full_addr + (nbuf_1) * 8);
                }
                last_buf_1 = buf_1;
                last_par_1 = par_1;
            }
            mbarrier_wait_hint(pv_done_addr + (last_buf_1) * 8, last_par_1, 10000000);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float sink_term_1 = 0.0f;
            if (has_sink_row_1 != 0) {
                float _exp2_3 = approx_exp2(sink_log2_1 - m_run_1);
                sink_term_1 = _exp2_3;
            }
            float denom_1 = psum_run_1 + sink_term_1;
            float _rcp_9 = approx_rcp(denom_1);
            float norm_1 = ((denom_1 > 0.0f) ? _rcp_9 * output_scale_1 : 0.0f);
            float o_values_1[16];
            unsigned int packed_o_1[8];
            int stage_base_1 = smem_v_addr;
            if (warp_rows_valid_1 != 0) {
                tmem_ld_x16(&o_values_1[0], taddr + 96 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_32 = {norm_1, norm_1};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_1)[_ls], _scale2_32);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values_1[_ls] = o_values_1[_ls] * norm_1;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_1[_lp*2 + 0], o_values_1[_lp*2+1 + 0]));
                    packed_o_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 16384 + (row_1 * 128 + (64 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 16384 + (row_1 * 128 + (80 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 3])));
                tmem_ld_x16(&o_values_1[0], taddr + 96 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_33 = {norm_1, norm_1};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_1)[_ls], _scale2_33);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values_1[_ls] = o_values_1[_ls] * norm_1;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_1[_lp*2 + 0], o_values_1[_lp*2+1 + 0]));
                    packed_o_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 16384 + (row_1 * 128 + (96 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 16384 + (row_1 * 128 + (112 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 3])));
                tmem_ld_x16(&o_values_1[0], taddr + 96 + 32 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_34 = {norm_1, norm_1};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_1)[_ls], _scale2_34);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values_1[_ls] = o_values_1[_ls] * norm_1;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_1[_lp*2 + 0], o_values_1[_lp*2+1 + 0]));
                    packed_o_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 32768 + (row_1 * 128 + (0 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 32768 + (row_1 * 128 + (16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 3])));
                tmem_ld_x16(&o_values_1[0], taddr + 96 + 48 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_35 = {norm_1, norm_1};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_1)[_ls], _scale2_35);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values_1[_ls] = o_values_1[_ls] * norm_1;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_1[_lp*2 + 0], o_values_1[_lp*2+1 + 0]));
                    packed_o_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 32768 + (row_1 * 128 + (32 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 32768 + (row_1 * 128 + (48 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 3])));
                tmem_ld_x16(&o_values_1[0], taddr + 96 + 64 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_36 = {norm_1, norm_1};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_1)[_ls], _scale2_36);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values_1[_ls] = o_values_1[_ls] * norm_1;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_1[_lp*2 + 0], o_values_1[_lp*2+1 + 0]));
                    packed_o_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 32768 + (row_1 * 128 + (64 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 32768 + (row_1 * 128 + (80 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 3])));
                tmem_ld_x16(&o_values_1[0], taddr + 96 + 80 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_37 = {norm_1, norm_1};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_1)[_ls], _scale2_37);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values_1[_ls] = o_values_1[_ls] * norm_1;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_1[_lp*2 + 0], o_values_1[_lp*2+1 + 0]));
                    packed_o_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 32768 + (row_1 * 128 + (96 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_1 + 32768 + (row_1 * 128 + (112 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_1[(4) + 3])));
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
    // ---- Role: compute2 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // compute2_main
            const int local_warp_2 = warp - 8;
            int rank_2 = cta_rank;
            int work_idx_2 = blockIdx.x / 2;
            int head_tile_2 = work_idx_2 % num_head_tiles;
            int split_work_2 = work_idx_2 / num_head_tiles;
            int split_idx_2 = split_work_2 % num_splits;
            int query_idx_2 = split_work_2 / num_splits;
            int head_base_2 = head_tile_2 * 128;
            const int row_2 = local_warp_2 * 32 + lane;
            int head_row_2 = head_base_2 + row_2;
            int row_valid_2 = ((head_row_2 < num_heads) ? 1 : 0);
            int warp_rows_valid_2 = ((head_base_2 + local_warp_2 * 32 < num_heads) ? 1 : 0);
            const int tmem_row_origin_2 = local_warp_2 * 32;
            int tile_lo_2 = split_idx_2 * tiles_per_split;
            int tile_hi_2 = tile_lo_2 + tiles_per_split;
            if (tile_hi_2 > total_tiles) {
                tile_hi_2 = total_tiles;
            }
            float softmax_scale_log2_2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_2 = bmm2_scale[0];
            float sink_log2_2 = 0.0f;
            int has_sink_row_2 = 0;
            if (has_sinks != 0 && split_idx_2 == 0 && row_valid_2 != 0) {
                has_sink_row_2 = 1;
                sink_log2_2 = sinks[head_row_2] * 1.4426950408889634f;
            }
            float inv_six_2 = 0.16666666666666666f;
            unsigned int _phase_q_nope_full0_0_2 = 0;
            mbarrier_wait_hint(q_nope_full0_addr, _phase_q_nope_full0_0_2, 10000000);
            _phase_q_nope_full0_0_2 ^= 1;
            unsigned int _phase_q_nope_full1_0_2 = 0;
            mbarrier_wait_hint(q_nope_full1_addr, _phase_q_nope_full1_0_2, 10000000);
            _phase_q_nope_full1_0_2 ^= 1;
            unsigned int _phase_q_nope_full2_0_2 = 0;
            mbarrier_wait_hint(q_nope_full2_addr, _phase_q_nope_full2_0_2, 10000000);
            _phase_q_nope_full2_0_2 ^= 1;
            const int q_warp_2 = warp;
            int unit_2 = 0;
            int q_block_2 = 0;
            int kset_2 = 0;
            int q_row_2 = 0;
            int q_row_addr_2 = 0;
            for (int i_2 = 0; i_2 < 3; i_2++) {
                unit_2 = q_warp_2 + 12 * i_2;
                if (unit_2 < 28) {
                    q_block_2 = ((unit_2 < 12) ? unit_2 % 4 : (unit_2 - 12) % 4);
                    kset_2 = ((unit_2 < 12) ? 4 + unit_2 / 4 : (unit_2 - 12) / 4);
                    q_row_2 = q_block_2 * 32 + lane;
                    if (head_base_2 + q_block_2 * 32 < num_heads) {
                        q_row_addr_2 = smem_qstage_addr + (unsigned int)(kset_2 * 16384) + (unsigned int)(q_row_2 * 128);
                        unsigned int words_2[8];
                        unsigned int sf_word_6 = 0;
                        unsigned int qa_2[4];
                        unsigned int qb_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 3]))
                            : "r"(q_row_addr_2 + (0 ^ q_row_2 % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 3]))
                            : "r"(q_row_addr_2 + (1 ^ q_row_2 % 8) * 16));
                        float qv_3[16];
                        qv_3[0] = __uint_as_float(qa_2[0] << 16);
                        qv_3[1] = __uint_as_float(qa_2[0] & 4294901760u);
                        qv_3[8] = __uint_as_float(qb_3[0] << 16);
                        qv_3[9] = __uint_as_float(qb_3[0] & 4294901760u);
                        qv_3[2] = __uint_as_float(qa_2[1] << 16);
                        qv_3[3] = __uint_as_float(qa_2[1] & 4294901760u);
                        qv_3[10] = __uint_as_float(qb_3[1] << 16);
                        qv_3[11] = __uint_as_float(qb_3[1] & 4294901760u);
                        qv_3[4] = __uint_as_float(qa_2[2] << 16);
                        qv_3[5] = __uint_as_float(qa_2[2] & 4294901760u);
                        qv_3[12] = __uint_as_float(qb_3[2] << 16);
                        qv_3[13] = __uint_as_float(qb_3[2] & 4294901760u);
                        qv_3[6] = __uint_as_float(qa_2[3] << 16);
                        qv_3[7] = __uint_as_float(qa_2[3] & 4294901760u);
                        qv_3[14] = __uint_as_float(qb_3[3] << 16);
                        qv_3[15] = __uint_as_float(qb_3[3] & 4294901760u);
                        float m8_2[8];
                        float _fabs_128 = fabsf(qv_3[0]);
                        float _fabs_129 = fabsf(qv_3[1]);
                        float _max_134 = max_noftz(_fabs_128, _fabs_129);
                        m8_2[0] = _max_134;
                        float _fabs_130 = fabsf(qv_3[2]);
                        float _fabs_131 = fabsf(qv_3[3]);
                        float _max_135 = max_noftz(_fabs_130, _fabs_131);
                        m8_2[1] = _max_135;
                        float _fabs_132 = fabsf(qv_3[4]);
                        float _fabs_133 = fabsf(qv_3[5]);
                        float _max_136 = max_noftz(_fabs_132, _fabs_133);
                        m8_2[2] = _max_136;
                        float _fabs_134 = fabsf(qv_3[6]);
                        float _fabs_135 = fabsf(qv_3[7]);
                        float _max_137 = max_noftz(_fabs_134, _fabs_135);
                        m8_2[3] = _max_137;
                        float _fabs_136 = fabsf(qv_3[8]);
                        float _fabs_137 = fabsf(qv_3[9]);
                        float _max_138 = max_noftz(_fabs_136, _fabs_137);
                        m8_2[4] = _max_138;
                        float _fabs_138 = fabsf(qv_3[10]);
                        float _fabs_139 = fabsf(qv_3[11]);
                        float _max_139 = max_noftz(_fabs_138, _fabs_139);
                        m8_2[5] = _max_139;
                        float _fabs_140 = fabsf(qv_3[12]);
                        float _fabs_141 = fabsf(qv_3[13]);
                        float _max_140 = max_noftz(_fabs_140, _fabs_141);
                        m8_2[6] = _max_140;
                        float _fabs_142 = fabsf(qv_3[14]);
                        float _fabs_143 = fabsf(qv_3[15]);
                        float _max_141 = max_noftz(_fabs_142, _fabs_143);
                        m8_2[7] = _max_141;
                        float m4_2[4];
                        float _max_142 = max_noftz(m8_2[0], m8_2[1]);
                        m4_2[0] = _max_142;
                        float _max_143 = max_noftz(m8_2[2], m8_2[3]);
                        m4_2[1] = _max_143;
                        float _max_144 = max_noftz(m8_2[4], m8_2[5]);
                        m4_2[2] = _max_144;
                        float _max_145 = max_noftz(m8_2[6], m8_2[7]);
                        m4_2[3] = _max_145;
                        float _max_146 = max_noftz(m4_2[0], m4_2[1]);
                        float _max_147 = max_noftz(m4_2[2], m4_2[3]);
                        float _max_148 = max_noftz(_max_146, _max_147);
                        float amax_2 = _max_148;
                        float sc_2 = amax_2 * inv_six_2;
                        uint16_t _e4m3x2_f32_40;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_40) : "f"(0.0f), "f"(sc_2));
                        uint16_t sc_pair_2 = _e4m3x2_f32_40;
                        unsigned int sc_byte_2 = (unsigned int)sc_pair_2 & 255;
                        unsigned int sc_exp_2 = sc_byte_2 >> 3 & 15;
                        unsigned int sc_man_2 = sc_byte_2 & 7;
                        float sc_norm_2 = __uint_as_float(sc_exp_2 + 120 << 23 | sc_man_2 << 20);
                        float sc_sub_2 = (float)sc_man_2 * 0.001953125f;
                        float sc_dec_2 = ((sc_exp_2 == 0) ? sc_sub_2 : sc_norm_2);
                        float _rcp_10 = __frcp_rn(sc_dec_2);
                        float inv_2 = ((sc_dec_2 > 0.0f) ? _rcp_10 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_0 = {inv_2, inv_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_3)[_ls], _scale2_0);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv_3[_ls] = qv_3[_ls] * inv_2;
                        }
                        #endif
                        uint32_t _fp4_pair_64;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_64) : "f"(qv_3[0]), "f"(qv_3[1]));
                        uint32_t _fp4_pair_65;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_65) : "f"(qv_3[2]), "f"(qv_3[3]));
                        uint32_t _fp4_pair_66;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_66) : "f"(qv_3[4]), "f"(qv_3[5]));
                        uint32_t _fp4_pair_67;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_67) : "f"(qv_3[6]), "f"(qv_3[7]));
                        uint32_t _fp4_pair_68;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_68) : "f"(qv_3[8]), "f"(qv_3[9]));
                        uint32_t _fp4_pair_69;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_69) : "f"(qv_3[10]), "f"(qv_3[11]));
                        uint32_t _fp4_pair_70;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_70) : "f"(qv_3[12]), "f"(qv_3[13]));
                        uint32_t _fp4_pair_71;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_71) : "f"(qv_3[14]), "f"(qv_3[15]));
                        words_2[0] = _fp4_pair_64 | _fp4_pair_65 << 8 | _fp4_pair_66 << 16 | _fp4_pair_67 << 24;
                        words_2[1] = _fp4_pair_68 | _fp4_pair_69 << 8 | _fp4_pair_70 << 16 | _fp4_pair_71 << 24;
                        sf_word_6 = sf_word_6 | sc_byte_2;
                        unsigned int qa_0_2[4];
                        unsigned int qb_1_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 3]))
                            : "r"(q_row_addr_2 + (2 ^ q_row_2 % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 3]))
                            : "r"(q_row_addr_2 + (3 ^ q_row_2 % 8) * 16));
                        float qv_2_2[16];
                        qv_2_2[0] = __uint_as_float(qa_0_2[0] << 16);
                        qv_2_2[1] = __uint_as_float(qa_0_2[0] & 4294901760u);
                        qv_2_2[8] = __uint_as_float(qb_1_2[0] << 16);
                        qv_2_2[9] = __uint_as_float(qb_1_2[0] & 4294901760u);
                        qv_2_2[2] = __uint_as_float(qa_0_2[1] << 16);
                        qv_2_2[3] = __uint_as_float(qa_0_2[1] & 4294901760u);
                        qv_2_2[10] = __uint_as_float(qb_1_2[1] << 16);
                        qv_2_2[11] = __uint_as_float(qb_1_2[1] & 4294901760u);
                        qv_2_2[4] = __uint_as_float(qa_0_2[2] << 16);
                        qv_2_2[5] = __uint_as_float(qa_0_2[2] & 4294901760u);
                        qv_2_2[12] = __uint_as_float(qb_1_2[2] << 16);
                        qv_2_2[13] = __uint_as_float(qb_1_2[2] & 4294901760u);
                        qv_2_2[6] = __uint_as_float(qa_0_2[3] << 16);
                        qv_2_2[7] = __uint_as_float(qa_0_2[3] & 4294901760u);
                        qv_2_2[14] = __uint_as_float(qb_1_2[3] << 16);
                        qv_2_2[15] = __uint_as_float(qb_1_2[3] & 4294901760u);
                        float m8_3_2[8];
                        float _fabs_144 = fabsf(qv_2_2[0]);
                        float _fabs_145 = fabsf(qv_2_2[1]);
                        float _max_149 = max_noftz(_fabs_144, _fabs_145);
                        m8_3_2[0] = _max_149;
                        float _fabs_146 = fabsf(qv_2_2[2]);
                        float _fabs_147 = fabsf(qv_2_2[3]);
                        float _max_150 = max_noftz(_fabs_146, _fabs_147);
                        m8_3_2[1] = _max_150;
                        float _fabs_148 = fabsf(qv_2_2[4]);
                        float _fabs_149 = fabsf(qv_2_2[5]);
                        float _max_151 = max_noftz(_fabs_148, _fabs_149);
                        m8_3_2[2] = _max_151;
                        float _fabs_150 = fabsf(qv_2_2[6]);
                        float _fabs_151 = fabsf(qv_2_2[7]);
                        float _max_152 = max_noftz(_fabs_150, _fabs_151);
                        m8_3_2[3] = _max_152;
                        float _fabs_152 = fabsf(qv_2_2[8]);
                        float _fabs_153 = fabsf(qv_2_2[9]);
                        float _max_153 = max_noftz(_fabs_152, _fabs_153);
                        m8_3_2[4] = _max_153;
                        float _fabs_154 = fabsf(qv_2_2[10]);
                        float _fabs_155 = fabsf(qv_2_2[11]);
                        float _max_154 = max_noftz(_fabs_154, _fabs_155);
                        m8_3_2[5] = _max_154;
                        float _fabs_156 = fabsf(qv_2_2[12]);
                        float _fabs_157 = fabsf(qv_2_2[13]);
                        float _max_155 = max_noftz(_fabs_156, _fabs_157);
                        m8_3_2[6] = _max_155;
                        float _fabs_158 = fabsf(qv_2_2[14]);
                        float _fabs_159 = fabsf(qv_2_2[15]);
                        float _max_156 = max_noftz(_fabs_158, _fabs_159);
                        m8_3_2[7] = _max_156;
                        float m4_4_2[4];
                        float _max_157 = max_noftz(m8_3_2[0], m8_3_2[1]);
                        m4_4_2[0] = _max_157;
                        float _max_158 = max_noftz(m8_3_2[2], m8_3_2[3]);
                        m4_4_2[1] = _max_158;
                        float _max_159 = max_noftz(m8_3_2[4], m8_3_2[5]);
                        m4_4_2[2] = _max_159;
                        float _max_160 = max_noftz(m8_3_2[6], m8_3_2[7]);
                        m4_4_2[3] = _max_160;
                        float _max_161 = max_noftz(m4_4_2[0], m4_4_2[1]);
                        float _max_162 = max_noftz(m4_4_2[2], m4_4_2[3]);
                        float _max_163 = max_noftz(_max_161, _max_162);
                        float amax_5_2 = _max_163;
                        float sc_6_2 = amax_5_2 * inv_six_2;
                        uint16_t _e4m3x2_f32_41;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_41) : "f"(0.0f), "f"(sc_6_2));
                        uint16_t sc_pair_7_2 = _e4m3x2_f32_41;
                        unsigned int sc_byte_8_2 = (unsigned int)sc_pair_7_2 & 255;
                        unsigned int sc_exp_9_2 = sc_byte_8_2 >> 3 & 15;
                        unsigned int sc_man_10_2 = sc_byte_8_2 & 7;
                        float sc_norm_11_2 = __uint_as_float(sc_exp_9_2 + 120 << 23 | sc_man_10_2 << 20);
                        float sc_sub_12_2 = (float)sc_man_10_2 * 0.001953125f;
                        float sc_dec_13_2 = ((sc_exp_9_2 == 0) ? sc_sub_12_2 : sc_norm_11_2);
                        float _rcp_11 = __frcp_rn(sc_dec_13_2);
                        float inv_14_2 = ((sc_dec_13_2 > 0.0f) ? _rcp_11 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_1 = {inv_14_2, inv_14_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2_2)[_ls], _scale2_1);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv_2_2[_ls] = qv_2_2[_ls] * inv_14_2;
                        }
                        #endif
                        uint32_t _fp4_pair_72;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_72) : "f"(qv_2_2[0]), "f"(qv_2_2[1]));
                        uint32_t _fp4_pair_73;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_73) : "f"(qv_2_2[2]), "f"(qv_2_2[3]));
                        uint32_t _fp4_pair_74;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_74) : "f"(qv_2_2[4]), "f"(qv_2_2[5]));
                        uint32_t _fp4_pair_75;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_75) : "f"(qv_2_2[6]), "f"(qv_2_2[7]));
                        uint32_t _fp4_pair_76;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_76) : "f"(qv_2_2[8]), "f"(qv_2_2[9]));
                        uint32_t _fp4_pair_77;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_77) : "f"(qv_2_2[10]), "f"(qv_2_2[11]));
                        uint32_t _fp4_pair_78;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_78) : "f"(qv_2_2[12]), "f"(qv_2_2[13]));
                        uint32_t _fp4_pair_79;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_79) : "f"(qv_2_2[14]), "f"(qv_2_2[15]));
                        words_2[2] = _fp4_pair_72 | _fp4_pair_73 << 8 | _fp4_pair_74 << 16 | _fp4_pair_75 << 24;
                        words_2[3] = _fp4_pair_76 | _fp4_pair_77 << 8 | _fp4_pair_78 << 16 | _fp4_pair_79 << 24;
                        sf_word_6 = sf_word_6 | sc_byte_8_2 << 8;
                        unsigned int qa_15_2[4];
                        unsigned int qb_16_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa_15_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_15_2[(0) + 3]))
                            : "r"(q_row_addr_2 + (4 ^ q_row_2 % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb_16_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_16_2[(0) + 3]))
                            : "r"(q_row_addr_2 + (5 ^ q_row_2 % 8) * 16));
                        float qv_17_2[16];
                        qv_17_2[0] = __uint_as_float(qa_15_2[0] << 16);
                        qv_17_2[1] = __uint_as_float(qa_15_2[0] & 4294901760u);
                        qv_17_2[8] = __uint_as_float(qb_16_2[0] << 16);
                        qv_17_2[9] = __uint_as_float(qb_16_2[0] & 4294901760u);
                        qv_17_2[2] = __uint_as_float(qa_15_2[1] << 16);
                        qv_17_2[3] = __uint_as_float(qa_15_2[1] & 4294901760u);
                        qv_17_2[10] = __uint_as_float(qb_16_2[1] << 16);
                        qv_17_2[11] = __uint_as_float(qb_16_2[1] & 4294901760u);
                        qv_17_2[4] = __uint_as_float(qa_15_2[2] << 16);
                        qv_17_2[5] = __uint_as_float(qa_15_2[2] & 4294901760u);
                        qv_17_2[12] = __uint_as_float(qb_16_2[2] << 16);
                        qv_17_2[13] = __uint_as_float(qb_16_2[2] & 4294901760u);
                        qv_17_2[6] = __uint_as_float(qa_15_2[3] << 16);
                        qv_17_2[7] = __uint_as_float(qa_15_2[3] & 4294901760u);
                        qv_17_2[14] = __uint_as_float(qb_16_2[3] << 16);
                        qv_17_2[15] = __uint_as_float(qb_16_2[3] & 4294901760u);
                        float m8_18_2[8];
                        float _fabs_160 = fabsf(qv_17_2[0]);
                        float _fabs_161 = fabsf(qv_17_2[1]);
                        float _max_164 = max_noftz(_fabs_160, _fabs_161);
                        m8_18_2[0] = _max_164;
                        float _fabs_162 = fabsf(qv_17_2[2]);
                        float _fabs_163 = fabsf(qv_17_2[3]);
                        float _max_165 = max_noftz(_fabs_162, _fabs_163);
                        m8_18_2[1] = _max_165;
                        float _fabs_164 = fabsf(qv_17_2[4]);
                        float _fabs_165 = fabsf(qv_17_2[5]);
                        float _max_166 = max_noftz(_fabs_164, _fabs_165);
                        m8_18_2[2] = _max_166;
                        float _fabs_166 = fabsf(qv_17_2[6]);
                        float _fabs_167 = fabsf(qv_17_2[7]);
                        float _max_167 = max_noftz(_fabs_166, _fabs_167);
                        m8_18_2[3] = _max_167;
                        float _fabs_168 = fabsf(qv_17_2[8]);
                        float _fabs_169 = fabsf(qv_17_2[9]);
                        float _max_168 = max_noftz(_fabs_168, _fabs_169);
                        m8_18_2[4] = _max_168;
                        float _fabs_170 = fabsf(qv_17_2[10]);
                        float _fabs_171 = fabsf(qv_17_2[11]);
                        float _max_169 = max_noftz(_fabs_170, _fabs_171);
                        m8_18_2[5] = _max_169;
                        float _fabs_172 = fabsf(qv_17_2[12]);
                        float _fabs_173 = fabsf(qv_17_2[13]);
                        float _max_170 = max_noftz(_fabs_172, _fabs_173);
                        m8_18_2[6] = _max_170;
                        float _fabs_174 = fabsf(qv_17_2[14]);
                        float _fabs_175 = fabsf(qv_17_2[15]);
                        float _max_171 = max_noftz(_fabs_174, _fabs_175);
                        m8_18_2[7] = _max_171;
                        float m4_19_2[4];
                        float _max_172 = max_noftz(m8_18_2[0], m8_18_2[1]);
                        m4_19_2[0] = _max_172;
                        float _max_173 = max_noftz(m8_18_2[2], m8_18_2[3]);
                        m4_19_2[1] = _max_173;
                        float _max_174 = max_noftz(m8_18_2[4], m8_18_2[5]);
                        m4_19_2[2] = _max_174;
                        float _max_175 = max_noftz(m8_18_2[6], m8_18_2[7]);
                        m4_19_2[3] = _max_175;
                        float _max_176 = max_noftz(m4_19_2[0], m4_19_2[1]);
                        float _max_177 = max_noftz(m4_19_2[2], m4_19_2[3]);
                        float _max_178 = max_noftz(_max_176, _max_177);
                        float amax_20_2 = _max_178;
                        float sc_21_2 = amax_20_2 * inv_six_2;
                        uint16_t _e4m3x2_f32_42;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_42) : "f"(0.0f), "f"(sc_21_2));
                        uint16_t sc_pair_22_2 = _e4m3x2_f32_42;
                        unsigned int sc_byte_23_2 = (unsigned int)sc_pair_22_2 & 255;
                        unsigned int sc_exp_24_2 = sc_byte_23_2 >> 3 & 15;
                        unsigned int sc_man_25_2 = sc_byte_23_2 & 7;
                        float sc_norm_26_2 = __uint_as_float(sc_exp_24_2 + 120 << 23 | sc_man_25_2 << 20);
                        float sc_sub_27_2 = (float)sc_man_25_2 * 0.001953125f;
                        float sc_dec_28_2 = ((sc_exp_24_2 == 0) ? sc_sub_27_2 : sc_norm_26_2);
                        float _rcp_12 = __frcp_rn(sc_dec_28_2);
                        float inv_29_2 = ((sc_dec_28_2 > 0.0f) ? _rcp_12 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_2 = {inv_29_2, inv_29_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_17_2)[_ls], _scale2_2);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv_17_2[_ls] = qv_17_2[_ls] * inv_29_2;
                        }
                        #endif
                        uint32_t _fp4_pair_80;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_80) : "f"(qv_17_2[0]), "f"(qv_17_2[1]));
                        uint32_t _fp4_pair_81;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_81) : "f"(qv_17_2[2]), "f"(qv_17_2[3]));
                        uint32_t _fp4_pair_82;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_82) : "f"(qv_17_2[4]), "f"(qv_17_2[5]));
                        uint32_t _fp4_pair_83;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_83) : "f"(qv_17_2[6]), "f"(qv_17_2[7]));
                        uint32_t _fp4_pair_84;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_84) : "f"(qv_17_2[8]), "f"(qv_17_2[9]));
                        uint32_t _fp4_pair_85;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_85) : "f"(qv_17_2[10]), "f"(qv_17_2[11]));
                        uint32_t _fp4_pair_86;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_86) : "f"(qv_17_2[12]), "f"(qv_17_2[13]));
                        uint32_t _fp4_pair_87;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_87) : "f"(qv_17_2[14]), "f"(qv_17_2[15]));
                        words_2[4] = _fp4_pair_80 | _fp4_pair_81 << 8 | _fp4_pair_82 << 16 | _fp4_pair_83 << 24;
                        words_2[5] = _fp4_pair_84 | _fp4_pair_85 << 8 | _fp4_pair_86 << 16 | _fp4_pair_87 << 24;
                        sf_word_6 = sf_word_6 | sc_byte_23_2 << 16;
                        unsigned int qa_30_2[4];
                        unsigned int qb_31_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qa_30_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_30_2[(0) + 3]))
                            : "r"(q_row_addr_2 + (6 ^ q_row_2 % 8) * 16));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qb_31_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_31_2[(0) + 3]))
                            : "r"(q_row_addr_2 + (7 ^ q_row_2 % 8) * 16));
                        float qv_32_2[16];
                        qv_32_2[0] = __uint_as_float(qa_30_2[0] << 16);
                        qv_32_2[1] = __uint_as_float(qa_30_2[0] & 4294901760u);
                        qv_32_2[8] = __uint_as_float(qb_31_2[0] << 16);
                        qv_32_2[9] = __uint_as_float(qb_31_2[0] & 4294901760u);
                        qv_32_2[2] = __uint_as_float(qa_30_2[1] << 16);
                        qv_32_2[3] = __uint_as_float(qa_30_2[1] & 4294901760u);
                        qv_32_2[10] = __uint_as_float(qb_31_2[1] << 16);
                        qv_32_2[11] = __uint_as_float(qb_31_2[1] & 4294901760u);
                        qv_32_2[4] = __uint_as_float(qa_30_2[2] << 16);
                        qv_32_2[5] = __uint_as_float(qa_30_2[2] & 4294901760u);
                        qv_32_2[12] = __uint_as_float(qb_31_2[2] << 16);
                        qv_32_2[13] = __uint_as_float(qb_31_2[2] & 4294901760u);
                        qv_32_2[6] = __uint_as_float(qa_30_2[3] << 16);
                        qv_32_2[7] = __uint_as_float(qa_30_2[3] & 4294901760u);
                        qv_32_2[14] = __uint_as_float(qb_31_2[3] << 16);
                        qv_32_2[15] = __uint_as_float(qb_31_2[3] & 4294901760u);
                        float m8_33_2[8];
                        float _fabs_176 = fabsf(qv_32_2[0]);
                        float _fabs_177 = fabsf(qv_32_2[1]);
                        float _max_179 = max_noftz(_fabs_176, _fabs_177);
                        m8_33_2[0] = _max_179;
                        float _fabs_178 = fabsf(qv_32_2[2]);
                        float _fabs_179 = fabsf(qv_32_2[3]);
                        float _max_180 = max_noftz(_fabs_178, _fabs_179);
                        m8_33_2[1] = _max_180;
                        float _fabs_180 = fabsf(qv_32_2[4]);
                        float _fabs_181 = fabsf(qv_32_2[5]);
                        float _max_181 = max_noftz(_fabs_180, _fabs_181);
                        m8_33_2[2] = _max_181;
                        float _fabs_182 = fabsf(qv_32_2[6]);
                        float _fabs_183 = fabsf(qv_32_2[7]);
                        float _max_182 = max_noftz(_fabs_182, _fabs_183);
                        m8_33_2[3] = _max_182;
                        float _fabs_184 = fabsf(qv_32_2[8]);
                        float _fabs_185 = fabsf(qv_32_2[9]);
                        float _max_183 = max_noftz(_fabs_184, _fabs_185);
                        m8_33_2[4] = _max_183;
                        float _fabs_186 = fabsf(qv_32_2[10]);
                        float _fabs_187 = fabsf(qv_32_2[11]);
                        float _max_184 = max_noftz(_fabs_186, _fabs_187);
                        m8_33_2[5] = _max_184;
                        float _fabs_188 = fabsf(qv_32_2[12]);
                        float _fabs_189 = fabsf(qv_32_2[13]);
                        float _max_185 = max_noftz(_fabs_188, _fabs_189);
                        m8_33_2[6] = _max_185;
                        float _fabs_190 = fabsf(qv_32_2[14]);
                        float _fabs_191 = fabsf(qv_32_2[15]);
                        float _max_186 = max_noftz(_fabs_190, _fabs_191);
                        m8_33_2[7] = _max_186;
                        float m4_34_2[4];
                        float _max_187 = max_noftz(m8_33_2[0], m8_33_2[1]);
                        m4_34_2[0] = _max_187;
                        float _max_188 = max_noftz(m8_33_2[2], m8_33_2[3]);
                        m4_34_2[1] = _max_188;
                        float _max_189 = max_noftz(m8_33_2[4], m8_33_2[5]);
                        m4_34_2[2] = _max_189;
                        float _max_190 = max_noftz(m8_33_2[6], m8_33_2[7]);
                        m4_34_2[3] = _max_190;
                        float _max_191 = max_noftz(m4_34_2[0], m4_34_2[1]);
                        float _max_192 = max_noftz(m4_34_2[2], m4_34_2[3]);
                        float _max_193 = max_noftz(_max_191, _max_192);
                        float amax_35_2 = _max_193;
                        float sc_36_2 = amax_35_2 * inv_six_2;
                        uint16_t _e4m3x2_f32_43;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_43) : "f"(0.0f), "f"(sc_36_2));
                        uint16_t sc_pair_37_2 = _e4m3x2_f32_43;
                        unsigned int sc_byte_38_2 = (unsigned int)sc_pair_37_2 & 255;
                        unsigned int sc_exp_39_2 = sc_byte_38_2 >> 3 & 15;
                        unsigned int sc_man_40_2 = sc_byte_38_2 & 7;
                        float sc_norm_41_2 = __uint_as_float(sc_exp_39_2 + 120 << 23 | sc_man_40_2 << 20);
                        float sc_sub_42_2 = (float)sc_man_40_2 * 0.001953125f;
                        float sc_dec_43_2 = ((sc_exp_39_2 == 0) ? sc_sub_42_2 : sc_norm_41_2);
                        float _rcp_13 = __frcp_rn(sc_dec_43_2);
                        float inv_44_2 = ((sc_dec_43_2 > 0.0f) ? _rcp_13 : 0.0f);
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_3 = {inv_44_2, inv_44_2};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_32_2)[_ls], _scale2_3);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++) {
                            qv_32_2[_ls] = qv_32_2[_ls] * inv_44_2;
                        }
                        #endif
                        uint32_t _fp4_pair_88;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_88) : "f"(qv_32_2[0]), "f"(qv_32_2[1]));
                        uint32_t _fp4_pair_89;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_89) : "f"(qv_32_2[2]), "f"(qv_32_2[3]));
                        uint32_t _fp4_pair_90;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_90) : "f"(qv_32_2[4]), "f"(qv_32_2[5]));
                        uint32_t _fp4_pair_91;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_91) : "f"(qv_32_2[6]), "f"(qv_32_2[7]));
                        uint32_t _fp4_pair_92;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_92) : "f"(qv_32_2[8]), "f"(qv_32_2[9]));
                        uint32_t _fp4_pair_93;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_93) : "f"(qv_32_2[10]), "f"(qv_32_2[11]));
                        uint32_t _fp4_pair_94;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_94) : "f"(qv_32_2[12]), "f"(qv_32_2[13]));
                        uint32_t _fp4_pair_95;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_95) : "f"(qv_32_2[14]), "f"(qv_32_2[15]));
                        words_2[6] = _fp4_pair_88 | _fp4_pair_89 << 8 | _fp4_pair_90 << 16 | _fp4_pair_91 << 24;
                        words_2[7] = _fp4_pair_92 | _fp4_pair_93 << 8 | _fp4_pair_94 << 16 | _fp4_pair_95 << 24;
                        sf_word_6 = sf_word_6 | sc_byte_38_2 << 24;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_qf4_addr + (unsigned int)(2 * kset_2 / 8 * 16384 + (q_row_2 * 128 + (2 * kset_2 % 8 * 16 ^ q_row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words_2[0])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_qf4_addr + (unsigned int)((2 * kset_2 + 1) / 8 * 16384 + (q_row_2 * 128 + ((2 * kset_2 + 1) % 8 * 16 ^ q_row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words_2[4])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(4) + 3])));
                        smem_qsf32[kset_2 / 4 * 2048 + q_row_2 % 32 / 8 * 512 + kset_2 % 4 * 128 + q_row_2 % 8 * 16 + q_row_2 / 32 % 4 * 4 >> 2] = sf_word_6;
                    } else {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(2 * kset_2 / 8 * 16384 + (q_row_2 * 128 + (2 * kset_2 % 8 * 16 ^ q_row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)((2 * kset_2 + 1) / 8 * 16384 + (q_row_2 * 128 + ((2 * kset_2 + 1) % 8 * 16 ^ q_row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        smem_qsf32[kset_2 / 4 * 2048 + q_row_2 % 32 / 8 * 512 + kset_2 % 4 * 128 + q_row_2 % 8 * 16 + q_row_2 / 32 % 4 * 4 >> 2] = 0;
                    }
                }
                if (i_2 == 0) {
                    mbarrier_arrive(q_ready_k_addr);
                }
            }
            {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(16384 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(16384 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                smem_qsf32[2048 + row_2 % 32 / 8 * 512 + 384 + row_2 % 8 * 16 + row_2 / 32 % 4 * 4 >> 2] = 0;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            unsigned int _phase_q_ready_0_2 = 0;
            mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0_2, 10000000);
            _phase_q_ready_0_2 ^= 1;
            float m_run_2 = -CAKE_INF;
            float l_run_2 = 0.0f;
            float psum_run_2 = 0.0f;
            unsigned int mask_words_2[4];
            int last_buf_2 = 0;
            int last_par_2 = 0;
            float sv0_2[16];
            float sv1_2[16];
            float sv2_2[16];
            unsigned int packed_p_2[4];
            unsigned int chunk_bits_2 = 0;
            float ov_2[16];
            int o_addr_2 = 0;
            float psum_v_2[1];
            mbarrier_wait_hint(tile_meta_addr, 0, 10000000);
            unsigned int my_mask_2 = smem_mask[local_warp_2];
            int valid_2 = (int)(my_mask_2 >> (unsigned int)lane & 1);
            mbarrier_wait_cluster_hint(kv_landed_addr, 0, 10000000);
            int kbase_2 = smem_v5_addr;
            int kzone_off_2 = 0;
            int vbase_2 = smem_v_addr;
            unsigned int zero4_2[4];
            zero4_2[0] = 0;
            zero4_2[1] = 0;
            zero4_2[2] = 0;
            zero4_2[3] = 0;
            int znope_hi_v_3 = 14;
            int zrope_pairs_hi_v_3 = 4;
            int nope_lo_2 = ((rank_2 == 0) ? 0 : 8);
            int nope_hi_v_2 = ((rank_2 == 0) ? 3 : 12);
            int rope_pair_lo_2 = 0;
            int rope_pairs_hi_v_2 = 0;
            {
                nope_lo_2 = ((rank_2 == 0) ? 6 : 14);
                nope_hi_v_2 = ((rank_2 == 0) ? 8 : 14);
                rope_pair_lo_2 = ((rank_2 == 0) ? 0 : 2);
                rope_pairs_hi_v_2 = ((rank_2 == 0) ? 0 : 4);
            }
            int vrank8_2 = 8 * rank_2;
            if (valid_2 != 0) {
                #pragma unroll 1
                for (int c_12 = nope_lo_2; c_12 < nope_hi_v_2; c_12++) {
                    unsigned int raw_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_4[(0) + 3]))
                        : "r"(kbase_2 + (c_12 / 8 * 16384 + (row_2 * 128 + (c_12 % 8 * 16 ^ row_2 % 8 * 16)))));
                    int sf_w_4 = c_12 / 2;
                    unsigned int sf_word_7 = smem_kzone32[kzone_off_2 + ((14 + (sf_w_4 >> 2)) / 8 * 16384 + (row_2 * 128 + ((14 + (sf_w_4 >> 2)) % 8 * 16 ^ row_2 % 8 * 16))) + 4 * (sf_w_4 & 3) >> 2];
                    int block_4 = 2 * c_12;
                    unsigned int scale_5 = sf_word_7 >> (unsigned int)(8 * ((c_12 & 1) * 2)) & 255;
                    unsigned int v8_5[4];
                    {
                        v8_5[0] = cake_dsv4_qmul4_portable<5>(raw_4[0], scale_5);
                    }
                    {
                        v8_5[1] = cake_dsv4_qmul4_portable<6>(raw_4[0], scale_5);
                    }
                    {
                        v8_5[2] = cake_dsv4_qmul4_portable<5>(raw_4[1], scale_5);
                    }
                    {
                        v8_5[3] = cake_dsv4_qmul4_portable<6>(raw_4[1], scale_5);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_2 + (c_12 - vrank8_2) / 4 * 16384 + (row_2 * 128 + (2 * c_12 % 8 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5[(0) + 3])));
                    int block_0_4 = 2 * c_12 + 1;
                    unsigned int scale_1_4 = sf_word_7 >> (unsigned int)(8 * ((c_12 & 1) * 2 + 1)) & 255;
                    unsigned int v8_2_4[4];
                    {
                        v8_2_4[0] = cake_dsv4_qmul4_portable<5>(raw_4[2], scale_1_4);
                    }
                    {
                        v8_2_4[1] = cake_dsv4_qmul4_portable<6>(raw_4[2], scale_1_4);
                    }
                    {
                        v8_2_4[2] = cake_dsv4_qmul4_portable<5>(raw_4[3], scale_1_4);
                    }
                    {
                        v8_2_4[3] = cake_dsv4_qmul4_portable<6>(raw_4[3], scale_1_4);
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_2 + (c_12 - vrank8_2) / 4 * 16384 + (row_2 * 128 + ((2 * c_12 + 1) % 8 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_2_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_4[(0) + 3])));
                }
                for (int jp_10 = rope_pair_lo_2; jp_10 < rope_pairs_hi_v_2; jp_10++) {
                    unsigned int vrope_4[4];
                    int j_4 = 2 * jp_10;
                    unsigned int rope_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_4[(0) + 3]))
                        : "r"(kbase_2 + 36864 + (row_2 * 128 + (j_4 * 16 ^ row_2 % 8 * 16))));
                    float lo_5 = __uint_as_float(rope_4[0] << 16);
                    float hi_6 = __uint_as_float(rope_4[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_44;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_44) : "f"(hi_6), "f"(lo_5));
                    uint16_t pair_6 = _e4m3x2_f32_44;
                    {
                        vrope_4[0] = (unsigned int)pair_6;
                    }
                    float lo_0_4 = __uint_as_float(rope_4[1] << 16);
                    float hi_1_4 = __uint_as_float(rope_4[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_45;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_45) : "f"(hi_1_4), "f"(lo_0_4));
                    uint16_t pair_2_4 = _e4m3x2_f32_45;
                    {
                        vrope_4[0] = vrope_4[0] | (unsigned int)pair_2_4 << 16;
                    }
                    float lo_3_4 = __uint_as_float(rope_4[2] << 16);
                    float hi_4_4 = __uint_as_float(rope_4[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_46;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_46) : "f"(hi_4_4), "f"(lo_3_4));
                    uint16_t pair_5_4 = _e4m3x2_f32_46;
                    {
                        vrope_4[1] = (unsigned int)pair_5_4;
                    }
                    float lo_6_4 = __uint_as_float(rope_4[3] << 16);
                    float hi_7_4 = __uint_as_float(rope_4[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_47;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_47) : "f"(hi_7_4), "f"(lo_6_4));
                    uint16_t pair_8_4 = _e4m3x2_f32_47;
                    {
                        vrope_4[1] = vrope_4[1] | (unsigned int)pair_8_4 << 16;
                    }
                    int j_9_4 = 2 * jp_10 + 1;
                    unsigned int rope_10_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_10_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_4[(0) + 3]))
                        : "r"(kbase_2 + 36864 + (row_2 * 128 + (j_9_4 * 16 ^ row_2 % 8 * 16))));
                    float lo_11_4 = __uint_as_float(rope_10_4[0] << 16);
                    float hi_12_4 = __uint_as_float(rope_10_4[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_48;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_48) : "f"(hi_12_4), "f"(lo_11_4));
                    uint16_t pair_13_4 = _e4m3x2_f32_48;
                    {
                        vrope_4[2] = (unsigned int)pair_13_4;
                    }
                    float lo_14_4 = __uint_as_float(rope_10_4[1] << 16);
                    float hi_15_4 = __uint_as_float(rope_10_4[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_49;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_49) : "f"(hi_15_4), "f"(lo_14_4));
                    uint16_t pair_16_4 = _e4m3x2_f32_49;
                    {
                        vrope_4[2] = vrope_4[2] | (unsigned int)pair_16_4 << 16;
                    }
                    float lo_17_4 = __uint_as_float(rope_10_4[2] << 16);
                    float hi_18_4 = __uint_as_float(rope_10_4[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_50;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_50) : "f"(hi_18_4), "f"(lo_17_4));
                    uint16_t pair_19_4 = _e4m3x2_f32_50;
                    {
                        vrope_4[3] = (unsigned int)pair_19_4;
                    }
                    float lo_20_4 = __uint_as_float(rope_10_4[3] << 16);
                    float hi_21_4 = __uint_as_float(rope_10_4[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_51;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_51) : "f"(hi_21_4), "f"(lo_20_4));
                    uint16_t pair_22_4 = _e4m3x2_f32_51;
                    {
                        vrope_4[3] = vrope_4[3] | (unsigned int)pair_22_4 << 16;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_2 + 16384 + (row_2 * 128 + ((4 + jp_10) * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&vrope_4[0])), "r"(*reinterpret_cast<uint32_t*>(&vrope_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vrope_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vrope_4[(0) + 3])));
                }
            } else {
                {
                    for (int jp_11 = 1; jp_11 < zrope_pairs_hi_v_3; jp_11++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(kbase_2 + 36864 + (row_2 * 128 + (2 * jp_11 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(kbase_2 + 36864 + (row_2 * 128 + ((2 * jp_11 + 1) * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 3])));
                    }
                }
                for (int c_13 = nope_lo_2; c_13 < nope_hi_v_2; c_13++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_2 + (c_13 - vrank8_2) / 4 * 16384 + (row_2 * 128 + (2 * c_13 % 8 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_2 + (c_13 - vrank8_2) / 4 * 16384 + (row_2 * 128 + ((2 * c_13 + 1) % 8 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 3])));
                }
                for (int jp_12 = rope_pair_lo_2; jp_12 < rope_pairs_hi_v_2; jp_12++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(vbase_2 + 16384 + (row_2 * 128 + ((4 + jp_12) * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_2[(0) + 3])));
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            #pragma unroll 1
            for (int t_2 = tile_lo_2; t_2 < tile_hi_2; t_2++) {
                int it_2 = t_2 - tile_lo_2;
                int buf_2 = it_2 & 1;
                int par_2 = it_2 >> 1 & 1;
                if (it_2 > 0) {
                    mbarrier_wait_hint(pv_done_addr + (last_buf_2) * 8, last_par_2, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(kbuf_free_p_addr + (last_buf_2) * 8);
                }
                mask_words_2[0] = smem_mask[(it_2 & 3) * 4];
                mask_words_2[1] = smem_mask[(it_2 & 3) * 4 + 1];
                mask_words_2[2] = smem_mask[(it_2 & 3) * 4 + 2];
                mask_words_2[3] = smem_mask[(it_2 & 3) * 4 + 3];
                kbase_2 = smem_v5_addr + (unsigned int)(buf_2 * 118784);
                mbarrier_wait_hint(s_full_addr + (buf_2) * 8, par_2, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp_rows_valid_2 != 0) {
                    tmem_ld_x16(&sv0_2[0], taddr + 256 + 96 + (unsigned int)(tmem_row_origin_2 << 16));
                    tmem_ld_x16(&sv1_2[0], taddr + 256 + 96 + 16 + (unsigned int)(tmem_row_origin_2 << 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                }
                mbarrier_arrive(s_free_addr);
                mbarrier_arrive(kbuf_free_k_addr + (buf_2) * 8);
                float slice_max_2 = -CAKE_INF;
                if (warp_rows_valid_2 != 0) {
                    chunk_bits_2 = mask_words_2[3] & 65535;
                    if (chunk_bits_2 != 65535) {
                        if ((chunk_bits_2 & 1) == 0) {
                            sv0_2[0] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 1 & 1) == 0) {
                            sv0_2[1] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 2 & 1) == 0) {
                            sv0_2[2] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 3 & 1) == 0) {
                            sv0_2[3] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 4 & 1) == 0) {
                            sv0_2[4] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 5 & 1) == 0) {
                            sv0_2[5] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 6 & 1) == 0) {
                            sv0_2[6] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 7 & 1) == 0) {
                            sv0_2[7] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 8 & 1) == 0) {
                            sv0_2[8] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 9 & 1) == 0) {
                            sv0_2[9] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 10 & 1) == 0) {
                            sv0_2[10] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 11 & 1) == 0) {
                            sv0_2[11] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 12 & 1) == 0) {
                            sv0_2[12] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 13 & 1) == 0) {
                            sv0_2[13] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 14 & 1) == 0) {
                            sv0_2[14] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 15 & 1) == 0) {
                            sv0_2[15] = -CAKE_INF;
                        }
                    }
                    float sv0_max_2 = sv0_2[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv0_max_2 = max_noftz(sv0_max_2, sv0_2[_lr]);
                    }
                    float _max_194 = max_noftz(slice_max_2, sv0_max_2);
                    slice_max_2 = _max_194;
                    chunk_bits_2 = mask_words_2[3] >> 16 & 65535;
                    if (chunk_bits_2 != 65535) {
                        if ((chunk_bits_2 & 1) == 0) {
                            sv1_2[0] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 1 & 1) == 0) {
                            sv1_2[1] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 2 & 1) == 0) {
                            sv1_2[2] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 3 & 1) == 0) {
                            sv1_2[3] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 4 & 1) == 0) {
                            sv1_2[4] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 5 & 1) == 0) {
                            sv1_2[5] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 6 & 1) == 0) {
                            sv1_2[6] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 7 & 1) == 0) {
                            sv1_2[7] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 8 & 1) == 0) {
                            sv1_2[8] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 9 & 1) == 0) {
                            sv1_2[9] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 10 & 1) == 0) {
                            sv1_2[10] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 11 & 1) == 0) {
                            sv1_2[11] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 12 & 1) == 0) {
                            sv1_2[12] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 13 & 1) == 0) {
                            sv1_2[13] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 14 & 1) == 0) {
                            sv1_2[14] = -CAKE_INF;
                        }
                        if ((chunk_bits_2 >> 15 & 1) == 0) {
                            sv1_2[15] = -CAKE_INF;
                        }
                    }
                    float sv1_max_2 = sv1_2[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv1_max_2 = max_noftz(sv1_max_2, sv1_2[_lr]);
                    }
                    float _max_195 = max_noftz(slice_max_2, sv1_max_2);
                    slice_max_2 = _max_195;
                }
                smem_pmax[256 + row_2] = slice_max_2;
                asm volatile("barrier.sync 8, 384;" ::: "memory");
                float _max_196 = max_noftz(smem_pmax[row_2], smem_pmax[128 + row_2]);
                float _max_197 = max_noftz(_max_196, smem_pmax[256 + row_2]);
                float tile_max_2 = _max_197;
                float cand_2 = tile_max_2 * softmax_scale_log2_2;
                if (it_2 == 0) {
                    if (has_sink_row_2 != 0) {
                        float _max_198 = max_noftz(cand_2, sink_log2_2);
                        cand_2 = _max_198;
                    }
                }
                float _max_199 = max_noftz(cand_2, m_run_2);
                cand_2 = _max_199;
                int grow_2 = 0;
                if (it_2 == 0) {
                    grow_2 = 1;
                }
                if (cand_2 - m_run_2 > 8.0f) {
                    grow_2 = 1;
                }
                if (grow_2 != 0) {
                    float _exp2_4 = approx_exp2(m_run_2 - cand_2);
                    float alpha_2 = ((m_run_2 > -CAKE_INF) ? _exp2_4 : 0.0f);
                    l_run_2 = l_run_2 * alpha_2;
                    psum_run_2 = psum_run_2 * alpha_2;
                    if (it_2 > 0) {
                        if (warp_rows_valid_2 != 0) {
                            o_addr_2 = taddr + 192 + (unsigned int)(tmem_row_origin_2 << 16);
                            tmem_ld_x16(&ov_2[0], o_addr_2);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_12 = {alpha_2, alpha_2};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_2)[_ls], _scale2_12);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov_2[_ls] = ov_2[_ls] * alpha_2;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr_2, ov_2);
                            o_addr_2 = taddr + 192 + 16 + (unsigned int)(tmem_row_origin_2 << 16);
                            tmem_ld_x16(&ov_2[0], o_addr_2);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_13 = {alpha_2, alpha_2};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_2)[_ls], _scale2_13);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov_2[_ls] = ov_2[_ls] * alpha_2;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr_2, ov_2);
                            o_addr_2 = taddr + 192 + 32 + (unsigned int)(tmem_row_origin_2 << 16);
                            tmem_ld_x16(&ov_2[0], o_addr_2);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_14 = {alpha_2, alpha_2};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_2)[_ls], _scale2_14);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov_2[_ls] = ov_2[_ls] * alpha_2;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr_2, ov_2);
                            o_addr_2 = taddr + 192 + 48 + (unsigned int)(tmem_row_origin_2 << 16);
                            tmem_ld_x16(&ov_2[0], o_addr_2);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_15 = {alpha_2, alpha_2};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ov_2)[_ls], _scale2_15);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                ov_2[_ls] = ov_2[_ls] * alpha_2;
                            }
                            #endif
                            tmem_st_x16_f32(o_addr_2, ov_2);
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                    }
                    m_run_2 = cand_2;
                }
                float m_use_2 = ((m_run_2 > -CAKE_INF) ? m_run_2 : 0.0f);
                float slice_sum_2 = 0.0f;
                if (warp_rows_valid_2 != 0) {
                    float score_bias_2 = -m_use_2;
                    const float2 _fma_b2_16 = {softmax_scale_log2_2, softmax_scale_log2_2};
                    const float2 _fma_c2_17 = {score_bias_2, score_bias_2};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv0_2)[_lf], _fma_b2_16, _fma_c2_17);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv0_2[_le] = approx_exp2(sv0_2[_le]);
                    }
                    float sv0_sum_2 = 0.0f;
                    #pragma unroll
                    for (int _lr = 0; _lr < 16; _lr++) {
                        sv0_sum_2 += sv0_2[_lr];
                    }
                    slice_sum_2 = slice_sum_2 + sv0_sum_2;
                    if (row_valid_2 == 0) {
                        sv0_2[0] = 0.0f;
                        sv0_2[1] = 0.0f;
                        sv0_2[2] = 0.0f;
                        sv0_2[3] = 0.0f;
                        sv0_2[4] = 0.0f;
                        sv0_2[5] = 0.0f;
                        sv0_2[6] = 0.0f;
                        sv0_2[7] = 0.0f;
                        sv0_2[8] = 0.0f;
                        sv0_2[9] = 0.0f;
                        sv0_2[10] = 0.0f;
                        sv0_2[11] = 0.0f;
                        sv0_2[12] = 0.0f;
                        sv0_2[13] = 0.0f;
                        sv0_2[14] = 0.0f;
                        sv0_2[15] = 0.0f;
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
                            : "=r"(_packed) : "f"(sv0_2[0]), "f"(sv0_2[1]),
                                               "f"(sv0_2[2]), "f"(sv0_2[3]));
                        packed_p_2[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv0_2[4]), "f"(sv0_2[5]),
                                               "f"(sv0_2[6]), "f"(sv0_2[7]));
                        packed_p_2[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv0_2[8]), "f"(sv0_2[9]),
                                               "f"(sv0_2[10]), "f"(sv0_2[11]));
                        packed_p_2[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv0_2[12]), "f"(sv0_2[13]),
                                               "f"(sv0_2[14]), "f"(sv0_2[15]));
                        packed_p_2[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(kbase_2 + 36864 + (row_2 * 128 + (96 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(0) + 3])));
                    const float2 _fma_b2_18 = {softmax_scale_log2_2, softmax_scale_log2_2};
                    const float2 _fma_c2_19 = {score_bias_2, score_bias_2};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv1_2)[_lf], _fma_b2_18, _fma_c2_19);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv1_2[_le] = approx_exp2(sv1_2[_le]);
                    }
                    float sv1_sum_2 = 0.0f;
                    #pragma unroll
                    for (int _lr = 0; _lr < 16; _lr++) {
                        sv1_sum_2 += sv1_2[_lr];
                    }
                    slice_sum_2 = slice_sum_2 + sv1_sum_2;
                    if (row_valid_2 == 0) {
                        sv1_2[0] = 0.0f;
                        sv1_2[1] = 0.0f;
                        sv1_2[2] = 0.0f;
                        sv1_2[3] = 0.0f;
                        sv1_2[4] = 0.0f;
                        sv1_2[5] = 0.0f;
                        sv1_2[6] = 0.0f;
                        sv1_2[7] = 0.0f;
                        sv1_2[8] = 0.0f;
                        sv1_2[9] = 0.0f;
                        sv1_2[10] = 0.0f;
                        sv1_2[11] = 0.0f;
                        sv1_2[12] = 0.0f;
                        sv1_2[13] = 0.0f;
                        sv1_2[14] = 0.0f;
                        sv1_2[15] = 0.0f;
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
                            : "=r"(_packed) : "f"(sv1_2[0]), "f"(sv1_2[1]),
                                               "f"(sv1_2[2]), "f"(sv1_2[3]));
                        packed_p_2[0] = _packed;
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
                            : "=r"(_packed) : "f"(sv1_2[4]), "f"(sv1_2[5]),
                                               "f"(sv1_2[6]), "f"(sv1_2[7]));
                        packed_p_2[1] = _packed;
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
                            : "=r"(_packed) : "f"(sv1_2[8]), "f"(sv1_2[9]),
                                               "f"(sv1_2[10]), "f"(sv1_2[11]));
                        packed_p_2[2] = _packed;
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
                            : "=r"(_packed) : "f"(sv1_2[12]), "f"(sv1_2[13]),
                                               "f"(sv1_2[14]), "f"(sv1_2[15]));
                        packed_p_2[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(kbase_2 + 36864 + (row_2 * 128 + (112 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(0) + 3])));
                }
                smem_psum[256 + row_2] = slice_sum_2;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr + (buf_2) * 8);
                asm volatile("barrier.sync 8, 384;" ::: "memory");
                l_run_2 = l_run_2 + smem_psum[row_2] + smem_psum[128 + row_2] + smem_psum[256 + row_2];
                mbarrier_wait_hint(psum_full_addr + (buf_2) * 8, par_2, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp_rows_valid_2 != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                        " {%0}, [%1];"
                        : "=f"(psum_v_2[0])
                        : "r"(taddr + 448 + (unsigned int)(tmem_row_origin_2 << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    psum_run_2 = psum_run_2 + psum_v_2[0];
                }
                if (tile_hi_2 > t_2 + 1) {
                    int nbuf_2 = buf_2 ^ 1;
                    int npar_2 = it_2 + 1 >> 1 & 1;
                    mbarrier_wait_hint(tile_meta_addr + (it_2 + 1 & 3) * 8, it_2 + 1 >> 2 & 1, 10000000);
                    my_mask_2 = smem_mask[(it_2 + 1 & 3) * 4 + local_warp_2];
                    valid_2 = (int)(my_mask_2 >> (unsigned int)lane & 1);
                    mbarrier_wait_cluster_hint(kv_landed_addr + (nbuf_2) * 8, npar_2, 10000000);
                    kbase_2 = smem_v5_addr + (unsigned int)(nbuf_2 * 118784);
                    kzone_off_2 = nbuf_2 * 118784;
                    vbase_2 = smem_v_addr + (unsigned int)(nbuf_2 * 32768);
                    unsigned int zero4_0_2[4];
                    zero4_0_2[0] = 0;
                    zero4_0_2[1] = 0;
                    zero4_0_2[2] = 0;
                    zero4_0_2[3] = 0;
                    int znope_hi_v_1_2 = 14;
                    int zrope_pairs_hi_v_2_2 = 4;
                    int nope_lo_3_2 = ((rank_2 == 0) ? 0 : 8);
                    int nope_hi_v_4_2 = ((rank_2 == 0) ? 3 : 12);
                    int rope_pair_lo_5_2 = 0;
                    int rope_pairs_hi_v_6_2 = 0;
                    {
                        nope_lo_3_2 = ((rank_2 == 0) ? 6 : 14);
                        nope_hi_v_4_2 = ((rank_2 == 0) ? 8 : 14);
                        rope_pair_lo_5_2 = ((rank_2 == 0) ? 0 : 2);
                        rope_pairs_hi_v_6_2 = ((rank_2 == 0) ? 0 : 4);
                    }
                    int vrank8_7_2 = 8 * rank_2;
                    if (valid_2 != 0) {
                        #pragma unroll 1
                        for (int c_14 = nope_lo_3_2; c_14 < nope_hi_v_4_2; c_14++) {
                            unsigned int raw_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_5[(0) + 3]))
                                : "r"(kbase_2 + (c_14 / 8 * 16384 + (row_2 * 128 + (c_14 % 8 * 16 ^ row_2 % 8 * 16)))));
                            int sf_w_5 = c_14 / 2;
                            unsigned int sf_word_8 = smem_kzone32[kzone_off_2 + ((14 + (sf_w_5 >> 2)) / 8 * 16384 + (row_2 * 128 + ((14 + (sf_w_5 >> 2)) % 8 * 16 ^ row_2 % 8 * 16))) + 4 * (sf_w_5 & 3) >> 2];
                            int block_5 = 2 * c_14;
                            unsigned int scale_6 = sf_word_8 >> (unsigned int)(8 * ((c_14 & 1) * 2)) & 255;
                            unsigned int v8_6[4];
                            {
                                v8_6[0] = cake_dsv4_qmul4_portable<5>(raw_5[0], scale_6);
                            }
                            {
                                v8_6[1] = cake_dsv4_qmul4_portable<6>(raw_5[0], scale_6);
                            }
                            {
                                v8_6[2] = cake_dsv4_qmul4_portable<5>(raw_5[1], scale_6);
                            }
                            {
                                v8_6[3] = cake_dsv4_qmul4_portable<6>(raw_5[1], scale_6);
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_2 + (c_14 - vrank8_7_2) / 4 * 16384 + (row_2 * 128 + (2 * c_14 % 8 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_6[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_6[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_6[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_6[(0) + 3])));
                            int block_0_5 = 2 * c_14 + 1;
                            unsigned int scale_1_5 = sf_word_8 >> (unsigned int)(8 * ((c_14 & 1) * 2 + 1)) & 255;
                            unsigned int v8_2_5[4];
                            {
                                v8_2_5[0] = cake_dsv4_qmul4_portable<5>(raw_5[2], scale_1_5);
                            }
                            {
                                v8_2_5[1] = cake_dsv4_qmul4_portable<6>(raw_5[2], scale_1_5);
                            }
                            {
                                v8_2_5[2] = cake_dsv4_qmul4_portable<5>(raw_5[3], scale_1_5);
                            }
                            {
                                v8_2_5[3] = cake_dsv4_qmul4_portable<6>(raw_5[3], scale_1_5);
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_2 + (c_14 - vrank8_7_2) / 4 * 16384 + (row_2 * 128 + ((2 * c_14 + 1) % 8 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_2_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_5[(0) + 3])));
                        }
                        for (int jp_13 = rope_pair_lo_5_2; jp_13 < rope_pairs_hi_v_6_2; jp_13++) {
                            unsigned int vrope_5[4];
                            int j_5 = 2 * jp_13;
                            unsigned int rope_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_5[(0) + 3]))
                                : "r"(kbase_2 + 36864 + (row_2 * 128 + (j_5 * 16 ^ row_2 % 8 * 16))));
                            float lo_7 = __uint_as_float(rope_5[0] << 16);
                            float hi_8 = __uint_as_float(rope_5[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_52;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_52) : "f"(hi_8), "f"(lo_7));
                            uint16_t pair_7 = _e4m3x2_f32_52;
                            {
                                vrope_5[0] = (unsigned int)pair_7;
                            }
                            float lo_0_5 = __uint_as_float(rope_5[1] << 16);
                            float hi_1_5 = __uint_as_float(rope_5[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_53;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_53) : "f"(hi_1_5), "f"(lo_0_5));
                            uint16_t pair_2_5 = _e4m3x2_f32_53;
                            {
                                vrope_5[0] = vrope_5[0] | (unsigned int)pair_2_5 << 16;
                            }
                            float lo_3_5 = __uint_as_float(rope_5[2] << 16);
                            float hi_4_5 = __uint_as_float(rope_5[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_54;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_54) : "f"(hi_4_5), "f"(lo_3_5));
                            uint16_t pair_5_5 = _e4m3x2_f32_54;
                            {
                                vrope_5[1] = (unsigned int)pair_5_5;
                            }
                            float lo_6_5 = __uint_as_float(rope_5[3] << 16);
                            float hi_7_5 = __uint_as_float(rope_5[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_55;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_55) : "f"(hi_7_5), "f"(lo_6_5));
                            uint16_t pair_8_5 = _e4m3x2_f32_55;
                            {
                                vrope_5[1] = vrope_5[1] | (unsigned int)pair_8_5 << 16;
                            }
                            int j_9_5 = 2 * jp_13 + 1;
                            unsigned int rope_10_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rope_10_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_5[(0) + 3]))
                                : "r"(kbase_2 + 36864 + (row_2 * 128 + (j_9_5 * 16 ^ row_2 % 8 * 16))));
                            float lo_11_5 = __uint_as_float(rope_10_5[0] << 16);
                            float hi_12_5 = __uint_as_float(rope_10_5[0] & 4294901760u);
                            uint16_t _e4m3x2_f32_56;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_56) : "f"(hi_12_5), "f"(lo_11_5));
                            uint16_t pair_13_5 = _e4m3x2_f32_56;
                            {
                                vrope_5[2] = (unsigned int)pair_13_5;
                            }
                            float lo_14_5 = __uint_as_float(rope_10_5[1] << 16);
                            float hi_15_5 = __uint_as_float(rope_10_5[1] & 4294901760u);
                            uint16_t _e4m3x2_f32_57;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_57) : "f"(hi_15_5), "f"(lo_14_5));
                            uint16_t pair_16_5 = _e4m3x2_f32_57;
                            {
                                vrope_5[2] = vrope_5[2] | (unsigned int)pair_16_5 << 16;
                            }
                            float lo_17_5 = __uint_as_float(rope_10_5[2] << 16);
                            float hi_18_5 = __uint_as_float(rope_10_5[2] & 4294901760u);
                            uint16_t _e4m3x2_f32_58;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_58) : "f"(hi_18_5), "f"(lo_17_5));
                            uint16_t pair_19_5 = _e4m3x2_f32_58;
                            {
                                vrope_5[3] = (unsigned int)pair_19_5;
                            }
                            float lo_20_5 = __uint_as_float(rope_10_5[3] << 16);
                            float hi_21_5 = __uint_as_float(rope_10_5[3] & 4294901760u);
                            uint16_t _e4m3x2_f32_59;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_59) : "f"(hi_21_5), "f"(lo_20_5));
                            uint16_t pair_22_5 = _e4m3x2_f32_59;
                            {
                                vrope_5[3] = vrope_5[3] | (unsigned int)pair_22_5 << 16;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_2 + 16384 + (row_2 * 128 + ((4 + jp_13) * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&vrope_5[0])), "r"(*reinterpret_cast<uint32_t*>(&vrope_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vrope_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vrope_5[(0) + 3])));
                        }
                    } else {
                        {
                            for (int jp_14 = 1; jp_14 < zrope_pairs_hi_v_2_2; jp_14++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                    "r"(kbase_2 + 36864 + (row_2 * 128 + (2 * jp_14 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 3])));
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                    "r"(kbase_2 + 36864 + (row_2 * 128 + ((2 * jp_14 + 1) * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 3])));
                            }
                        }
                        for (int c_15 = nope_lo_3_2; c_15 < nope_hi_v_4_2; c_15++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_2 + (c_15 - vrank8_7_2) / 4 * 16384 + (row_2 * 128 + (2 * c_15 % 8 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 3])));
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_2 + (c_15 - vrank8_7_2) / 4 * 16384 + (row_2 * 128 + ((2 * c_15 + 1) % 8 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 3])));
                        }
                        for (int jp_15 = rope_pair_lo_5_2; jp_15 < rope_pairs_hi_v_6_2; jp_15++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(vbase_2 + 16384 + (row_2 * 128 + ((4 + jp_15) * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4_0_2[(0) + 3])));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_full_addr + (nbuf_2) * 8);
                }
                last_buf_2 = buf_2;
                last_par_2 = par_2;
            }
            mbarrier_wait_hint(pv_done_addr + (last_buf_2) * 8, last_par_2, 10000000);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float sink_term_2 = 0.0f;
            if (has_sink_row_2 != 0) {
                float _exp2_5 = approx_exp2(sink_log2_2 - m_run_2);
                sink_term_2 = _exp2_5;
            }
            float denom_2 = psum_run_2 + sink_term_2;
            float _rcp_14 = approx_rcp(denom_2);
            float norm_2 = ((denom_2 > 0.0f) ? _rcp_14 * output_scale_2 : 0.0f);
            float o_values_2[16];
            unsigned int packed_o_2[8];
            int stage_base_2 = smem_v_addr;
            if (warp_rows_valid_2 != 0) {
                tmem_ld_x16(&o_values_2[0], taddr + 192 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_28 = {norm_2, norm_2};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_2)[_ls], _scale2_28);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values_2[_ls] = o_values_2[_ls] * norm_2;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_2[_lp*2 + 0], o_values_2[_lp*2+1 + 0]));
                    packed_o_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_2 + 49152 + (row_2 * 128 + (0 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_2 + 49152 + (row_2 * 128 + (16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 3])));
                tmem_ld_x16(&o_values_2[0], taddr + 192 + 16 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_29 = {norm_2, norm_2};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_2)[_ls], _scale2_29);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values_2[_ls] = o_values_2[_ls] * norm_2;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_2[_lp*2 + 0], o_values_2[_lp*2+1 + 0]));
                    packed_o_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_2 + 49152 + (row_2 * 128 + (32 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_2 + 49152 + (row_2 * 128 + (48 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 3])));
                tmem_ld_x16(&o_values_2[0], taddr + 192 + 32 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_30 = {norm_2, norm_2};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_2)[_ls], _scale2_30);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values_2[_ls] = o_values_2[_ls] * norm_2;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_2[_lp*2 + 0], o_values_2[_lp*2+1 + 0]));
                    packed_o_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_2 + 49152 + (row_2 * 128 + (64 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_2 + 49152 + (row_2 * 128 + (80 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 3])));
                tmem_ld_x16(&o_values_2[0], taddr + 192 + 48 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_31 = {norm_2, norm_2};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_2)[_ls], _scale2_31);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++) {
                    o_values_2[_ls] = o_values_2[_ls] * norm_2;
                }
                #endif
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_2[_lp*2 + 0], o_values_2[_lp*2+1 + 0]));
                    packed_o_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_2 + 49152 + (row_2 * 128 + (96 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(stage_base_2 + 49152 + (row_2 * 128 + (112 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_o_2[(4) + 3])));
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            mbarrier_arrive(tmem_dealloc_addr);
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            int m_work = blockIdx.x / 2;
            int m_split = m_work / num_head_tiles % num_splits;
            int m_tile_lo = m_split * tiles_per_split;
            int m_tile_hi = m_tile_lo + tiles_per_split;
            if (m_tile_hi > total_tiles) {
                m_tile_hi = total_tiles;
            }
            unsigned int _phase_q_ready_0_3 = 0;
            mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0_3, 10000000);
            _phase_q_ready_0_3 ^= 1;
            unsigned int _phase_q_rope_full_0 = 0;
            mbarrier_wait_hint(q_rope_full_addr, _phase_q_rope_full_0, 10000000);
            _phase_q_rope_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (elect_sync()) {
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa0, make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4))));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 4), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 8), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 12), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 24)));
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa1, make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 4), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 8), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 12), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 24)));
            }
            #pragma unroll 1
            for (int t_3 = m_tile_lo; t_3 < m_tile_hi; t_3++) {
                int it_3 = t_3 - m_tile_lo;
                int par_3 = it_3 >> 1 & 1;
                int first = ((it_3 == 0) ? 1 : 0);
                if ((it_3 & 1) == 0) {
                    mbarrier_wait_cluster(kv_landed_addr, par_3);
                    mbarrier_wait(k_ready_addr, par_3);
                    mbarrier_wait(s_free_addr, it_3 & 1 ^ 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb0, make_sf_cp_desc_lo_sbo512((((smem_v6_addr) >> 4))));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 4), make_sf_cp_desc_lo_sbo512((((smem_v6_addr) >> 4) + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 8), make_sf_cp_desc_lo_sbo512((((smem_v6_addr) >> 4) + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 12), make_sf_cp_desc_lo_sbo512((((smem_v6_addr) >> 4) + 24)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb1, make_sf_cp_desc_lo_sbo512((((smem_v6_addr) >> 4) + 128)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 4), make_sf_cp_desc_lo_sbo512((((smem_v6_addr) >> 4) + 128 + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 8), make_sf_cp_desc_lo_sbo512((((smem_v6_addr) >> 4) + 128 + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 12), make_sf_cp_desc_lo_sbo512((((smem_v6_addr) >> 4) + 128 + 24)));
                        int _mma_a_lo_0 = ((smem_qrope_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_0 = ((smem_v8_addr) >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_tmem_s), "r"(0));
                        int _mma_a_lo_1 = (((smem_qf4_addr) >> 4) & 0x3FFF) + (0) * 1024;
                        int _mma_b_lo_1 = (((smem_v5_addr) >> 4) & 0x3FFF) + (0) * 1024;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                0x8200480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb0 + 0, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                0x8200480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb0 + 4, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                0x8200480U, tmem_tmem_sfa0 + 8, tmem_tmem_sfb0 + 8, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                0x8200480U, tmem_tmem_sfa0 + 12, tmem_tmem_sfb0 + 12, 1);
                        }
                        int _mma_a_lo_2 = (((smem_qf4_addr) >> 4) & 0x3FFF) + (1) * 1024;
                        int _mma_b_lo_2 = (((smem_v5_addr) >> 4) & 0x3FFF) + (1) * 1024;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                0x8200480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb1 + 0, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                0x8200480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb1 + 4, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                0x8200480U, tmem_tmem_sfa1 + 8, tmem_tmem_sfb1 + 8, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                0x8200480U, tmem_tmem_sfa1 + 12, tmem_tmem_sfb1 + 12, 1);
                        }
                        tcgen05_commit(s_full_addr);
                    }
                    mbarrier_wait(p_full_addr, par_3);
                    mbarrier_wait(kv_full_addr, par_3);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_3 = ((smem_v9_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_3 = ((smem_ones_addr) >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_tmem_psum), "r"(0));
                        tcgen05_commit(psum_full_addr);
                        int _mma_b_lo_4 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (0) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_4), "r"(tmem_tmem_o0), "r"(((first) ? 0 : 1)));
                        int _mma_b_lo_5 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (1) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_5), "r"(tmem_tmem_o1), "r"(((first) ? 0 : 1)));
                        tcgen05_commit(pv_done_addr);
                    }
                } else {
                    mbarrier_wait_cluster(kv_landed_addr + 8, par_3);
                    mbarrier_wait(k_ready_addr + 8, par_3);
                    mbarrier_wait(s_free_addr, it_3 & 1 ^ 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb0, make_sf_cp_desc_lo_sbo512((((smem_v11_addr) >> 4))));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 4), make_sf_cp_desc_lo_sbo512((((smem_v11_addr) >> 4) + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 8), make_sf_cp_desc_lo_sbo512((((smem_v11_addr) >> 4) + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 12), make_sf_cp_desc_lo_sbo512((((smem_v11_addr) >> 4) + 24)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb1, make_sf_cp_desc_lo_sbo512((((smem_v11_addr) >> 4) + 128)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 4), make_sf_cp_desc_lo_sbo512((((smem_v11_addr) >> 4) + 128 + 8)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 8), make_sf_cp_desc_lo_sbo512((((smem_v11_addr) >> 4) + 128 + 16)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 12), make_sf_cp_desc_lo_sbo512((((smem_v11_addr) >> 4) + 128 + 24)));
                        int _mma_a_lo_6 = ((smem_qrope_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_6 = ((smem_v13_addr) >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_tmem_s), "r"(0));
                        int _mma_a_lo_7 = (((smem_qf4_addr) >> 4) & 0x3FFF) + (0) * 1024;
                        int _mma_b_lo_7 = (((smem_v10_addr) >> 4) & 0x3FFF) + (0) * 1024;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                0x8200480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb0 + 0, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                0x8200480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb0 + 4, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                0x8200480U, tmem_tmem_sfa0 + 8, tmem_tmem_sfb0 + 8, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                0x8200480U, tmem_tmem_sfa0 + 12, tmem_tmem_sfb0 + 12, 1);
                        }
                        int _mma_a_lo_8 = (((smem_qf4_addr) >> 4) & 0x3FFF) + (1) * 1024;
                        int _mma_b_lo_8 = (((smem_v10_addr) >> 4) & 0x3FFF) + (1) * 1024;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_8) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_8) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                                0x8200480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb1 + 0, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                                0x8200480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb1 + 4, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                                0x8200480U, tmem_tmem_sfa1 + 8, tmem_tmem_sfb1 + 8, 1);
                            tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                                0x8200480U, tmem_tmem_sfa1 + 12, tmem_tmem_sfb1 + 12, 1);
                        }
                        tcgen05_commit(s_full_addr + 8);
                    }
                    mbarrier_wait(p_full_addr + 8, par_3);
                    mbarrier_wait(kv_full_addr + 8, par_3);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_9 = ((smem_v14_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_9 = ((smem_ones_addr) >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134479888;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"(tmem_tmem_psum), "r"(0));
                        tcgen05_commit(psum_full_addr + 8);
                        int _mma_b_lo_10 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (2) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_10), "r"(tmem_tmem_o0), "r"(((first) ? 0 : 1)));
                        int _mma_b_lo_11 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (3) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_11), "r"(tmem_tmem_o1), "r"(((first) ? 0 : 1)));
                        tcgen05_commit(pv_done_addr + 8);
                    }
                }
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait_hint(tmem_dealloc_addr, _phase_tmem_dealloc_0, 10000000);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 13 && warp <= 15) {
        { // load_warp_main
            const int load_tid = (warp - 13) * 32 + lane;
            const int load_warp_idx = warp - 13;
            int rank_3 = cta_rank;
            int work_idx_3 = blockIdx.x / 2;
            int head_tile_3 = work_idx_3 % num_head_tiles;
            int split_work_3 = work_idx_3 / num_head_tiles;
            int split_idx_3 = split_work_3 % num_splits;
            int query_idx_3 = split_work_3 / num_splits;
            int tile_lo_3 = split_idx_3 * tiles_per_split;
            int tile_hi_3 = tile_lo_3 + tiles_per_split;
            if (tile_hi_3 > total_tiles) {
                tile_hi_3 = total_tiles;
            }
            if (load_tid == 0) {
                mbarrier_arrive_expect_tx(q_nope_full0_addr, 49152);
                tma_4d_gmem2smem(smem_qstage_addr, (&tmap_q), 0, head_tile_3 * 128, 0, query_idx_3, q_nope_full0_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 16384, (&tmap_q), 0, head_tile_3 * 128, 1, query_idx_3, q_nope_full0_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 32768, (&tmap_q), 0, head_tile_3 * 128, 2, query_idx_3, q_nope_full0_addr);
                mbarrier_arrive_expect_tx(q_nope_full1_addr, 32768);
                tma_4d_gmem2smem(smem_qstage_addr + 49152, (&tmap_q), 0, head_tile_3 * 128, 3, query_idx_3, q_nope_full1_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 65536, (&tmap_q), 0, head_tile_3 * 128, 4, query_idx_3, q_nope_full1_addr);
                mbarrier_arrive_expect_tx(q_nope_full2_addr, 32768);
                tma_4d_gmem2smem(smem_qstage_addr + 81920, (&tmap_q), 0, head_tile_3 * 128, 5, query_idx_3, q_nope_full2_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 98304, (&tmap_q), 0, head_tile_3 * 128, 6, query_idx_3, q_nope_full2_addr);
                mbarrier_arrive_expect_tx(q_rope_full_addr, 16384);
                tma_4d_gmem2smem(smem_qrope_addr, (&tmap_q), 0, head_tile_3 * 128, 7, query_idx_3, q_rope_full_addr);
            }
            int rtid = load_tid - 32;
            int rwarp = load_warp_idx - 1;
            if (load_warp_idx >= 1) {
                int is_main = 1;
                if (tile_lo_3 >= num_main_tiles) {
                    is_main = 0;
                }
                int tile_in_table = ((is_main != 0) ? tile_lo_3 : tile_lo_3 - num_main_tiles);
                int table_width = ((is_main != 0) ? main_width : extra_width);
                int* row_ptr = ((is_main != 0) ? (main_indices + (query_idx_3 * main_index_stride)) : (extra_indices + (query_idx_3 * extra_index_stride)));
                int active_len = table_width;
                if (is_main != 0) {
                    if (has_main_lengths != 0) {
                        active_len = main_lengths[query_idx_3];
                    }
                } else if (has_extra_lengths != 0) {
                    active_len = extra_lengths[query_idx_3];
                }
                if (active_len < 0) {
                    active_len = 0;
                }
                if (active_len > table_width) {
                    active_len = table_width;
                }
                int col0 = tile_in_table * 128 + rtid;
                int col1 = col0 + 64;
                int raw0 = -1;
                int raw1 = -1;
                if (col0 < table_width) {
                    raw0 = row_ptr[col0];
                }
                if (col1 < table_width) {
                    raw1 = row_ptr[col1];
                }
                int valid0 = 1;
                if (raw0 < 0) {
                    valid0 = 0;
                }
                if (col0 >= active_len) {
                    valid0 = 0;
                }
                int valid1 = 1;
                if (raw1 < 0) {
                    valid1 = 0;
                }
                if (col1 >= active_len) {
                    valid1 = 0;
                }
                int tok0 = ((valid0 != 0) ? raw0 : -1);
                int tok1 = ((valid1 != 0) ? raw1 : -1);
                smem_tok[rtid] = ((rank_3 == 0) ? tok0 : tok1);
                unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, valid0 != 0);
                unsigned int bits0 = _vote_0;
                unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, valid1 != 0);
                unsigned int bits1 = _vote_1;
                if (lane == 0) {
                    smem_mask[rwarp] = bits0;
                    smem_mask[2 + rwarp] = bits1;
                }
                asm volatile("barrier.sync 10, 64;" ::: "memory");
                mbarrier_arrive(tile_meta_addr);
            }
            int peer = rank_3 ^ 1;
            unsigned int _phase_q_ready_k_0 = 0;
            #pragma unroll 1
            for (int t_4 = tile_lo_3; t_4 < tile_hi_3; t_4++) {
                int it_4 = t_4 - tile_lo_3;
                int buf_3 = it_4 & 1;
                asm volatile("barrier.sync 9, 96;" ::: "memory");
                int is_main_1 = 1;
                if (t_4 >= num_main_tiles) {
                    is_main_1 = 0;
                }
                uint8_t* cache = ((is_main_1 != 0) ? (main_cache) : (extra_cache));
                int page_shift = ((is_main_1 != 0) ? main_page_shift : extra_page_shift);
                long long page_stride = ((is_main_1 != 0) ? main_page_stride : extra_page_stride);
                int page_size = 1 << page_shift;
                int kf4_addr = smem_v5_addr + (unsigned int)(buf_3 * 118784);
                if (it_4 >= 2) {
                    mbarrier_wait_hint(kbuf_free_k_addr + (buf_3) * 8, it_4 - 2 >> 1 & 1, 10000000);
                }
                if (it_4 == 1) {
                    mbarrier_wait_hint(q_ready_k_addr, _phase_q_ready_k_0, 10000000);
                    _phase_q_ready_k_0 ^= 1;
                }
                if (load_tid == 0) {
                    if (it_4 >= 1) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(peer_free_k_addr + buf_3 * 8), "r"(peer) : "memory");
                    }
                    mbarrier_arrive_expect_tx(kv_landed_addr + (buf_3) * 8, 24576);
                }
                if (load_tid < 64) {
                    int tok = smem_tok[buf_3 * 64 + load_tid];
                    int grow_3 = 0;
                    int page = 0;
                    int slot = 0;
                    int sw = 0;
                    long long base = 0;
                    long long gsrc = 0;
                    int dst = 0;
                    if (tok >= 0) {
                        grow_3 = rank_3 * 64 + load_tid;
                        page = tok >> page_shift;
                        slot = tok - (page << page_shift);
                        sw = (grow_3 & 7) * 16;
                        base = (long long)page * page_stride;
                        gsrc = base + (long long)(slot * 352);
                        dst = kf4_addr + grow_3 * 128;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst + (sw ^ 0)), "l"(cache + gsrc));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst + (sw ^ 16)), "l"(cache + (gsrc + 16)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst + (sw ^ 32)), "l"(cache + (gsrc + 32)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst + (sw ^ 48)), "l"(cache + (gsrc + 48)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst + (sw ^ 64)), "l"(cache + (gsrc + 64)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst + (sw ^ 80)), "l"(cache + (gsrc + 80)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst + (sw ^ 96)), "l"(cache + (gsrc + 96)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst + (sw ^ 112)), "l"(cache + (gsrc + 112)));
                    }
                } else {
                    int tok_1 = smem_tok[buf_3 * 64 + (load_tid - 64)];
                    int grow_4 = 0;
                    int page_1 = 0;
                    int slot_1 = 0;
                    int sw_1 = 0;
                    long long base_1 = 0;
                    long long gsrc_1 = 0;
                    int dst_1 = 0;
                    if (tok_1 >= 0) {
                        grow_4 = rank_3 * 64 + (load_tid - 64);
                        page_1 = tok_1 >> page_shift;
                        slot_1 = tok_1 - (page_1 << page_shift);
                        sw_1 = (grow_4 & 7) * 16;
                        base_1 = (long long)page_1 * page_stride;
                        gsrc_1 = base_1 + (long long)(slot_1 * 352);
                        dst_1 = kf4_addr + grow_4 * 128;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_1 + 16384 + (sw_1 ^ 0)), "l"(cache + (gsrc_1 + 128)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_1 + 16384 + (sw_1 ^ 16)), "l"(cache + (gsrc_1 + 144)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_1 + 16384 + (sw_1 ^ 32)), "l"(cache + (gsrc_1 + 160)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_1 + 16384 + (sw_1 ^ 48)), "l"(cache + (gsrc_1 + 176)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_1 + 16384 + (sw_1 ^ 64)), "l"(cache + (gsrc_1 + 192)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_1 + 16384 + (sw_1 ^ 80)), "l"(cache + (gsrc_1 + 208)));
                        gsrc_1 = base_1 + (long long)(page_size * 352 + slot_1 * 32);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_1 + 16384 + (sw_1 ^ 96)), "l"(cache + gsrc_1));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_1 + 16384 + (sw_1 ^ 112)), "l"(cache + (gsrc_1 + 16)));
                    }
                }
                if (load_tid < 32) {
                    int tok_2 = smem_tok[buf_3 * 64 + (load_tid + 32)];
                    int grow_5 = 0;
                    int page_2 = 0;
                    int slot_2 = 0;
                    int sw_2 = 0;
                    long long base_2 = 0;
                    long long gsrc_2 = 0;
                    int dst_2 = 0;
                    if (tok_2 >= 0) {
                        grow_5 = rank_3 * 64 + (load_tid + 32);
                        page_2 = tok_2 >> page_shift;
                        slot_2 = tok_2 - (page_2 << page_shift);
                        sw_2 = (grow_5 & 7) * 16;
                        base_2 = (long long)page_2 * page_stride;
                        gsrc_2 = base_2 + (long long)(slot_2 * 352);
                        dst_2 = kf4_addr + grow_5 * 128;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_2 + 16384 + (sw_2 ^ 0)), "l"(cache + (gsrc_2 + 128)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_2 + 16384 + (sw_2 ^ 16)), "l"(cache + (gsrc_2 + 144)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_2 + 16384 + (sw_2 ^ 32)), "l"(cache + (gsrc_2 + 160)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_2 + 16384 + (sw_2 ^ 48)), "l"(cache + (gsrc_2 + 176)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_2 + 16384 + (sw_2 ^ 64)), "l"(cache + (gsrc_2 + 192)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_2 + 16384 + (sw_2 ^ 80)), "l"(cache + (gsrc_2 + 208)));
                        gsrc_2 = base_2 + (long long)(page_size * 352 + slot_2 * 32);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_2 + 16384 + (sw_2 ^ 96)), "l"(cache + gsrc_2));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_2 + 16384 + (sw_2 ^ 112)), "l"(cache + (gsrc_2 + 16)));
                    }
                }
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(gather_landed_k_addr + (buf_3) * 8) : "memory");
                int slab_off = kf4_addr + rank_3 * 8192;
                if (load_warp_idx == 0) {
                    mbarrier_wait_hint(gather_landed_k_addr + (buf_3) * 8, it_4 >> 1 & 1, 10000000);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (it_4 >= 1) {
                        mbarrier_wait_cluster_hint(peer_free_k_addr + (buf_3) * 8, it_4 - 2 + buf_3 >> 1 & 1, 10000000);
                    }
                    if (lane == 0) {
                        uint32_t _mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_0) : "r"(kv_landed_addr + buf_3 * 8), "r"(peer));
                        uint32_t _mapa_1;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_1) : "r"(slab_off), "r"(peer));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_1), "r"(slab_off), "r"((uint32_t)(8192)), "r"(_mapa_0)
                            : "memory");
                        uint32_t _mapa_2;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_2) : "r"(slab_off + 16384), "r"(peer));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_2), "r"(slab_off + 16384), "r"((uint32_t)(8192)), "r"(_mapa_0)
                            : "memory");
                    }
                }
                if (it_4 >= 2) {
                    mbarrier_wait_hint(kbuf_free_p_addr + (buf_3) * 8, it_4 - 2 >> 1 & 1, 10000000);
                }
                if (it_4 >= 1) {
                    if (load_tid == 0) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(peer_free_p_addr + buf_3 * 8), "r"(peer) : "memory");
                    }
                }
                if (load_tid < 64) {
                    int tok_3 = smem_tok[buf_3 * 64 + load_tid];
                    int grow_6 = 0;
                    int page_3 = 0;
                    int slot_3 = 0;
                    int sw_3 = 0;
                    long long base_3 = 0;
                    long long gsrc_3 = 0;
                    int dst_3 = 0;
                    if (tok_3 >= 0) {
                        grow_6 = rank_3 * 64 + load_tid;
                        page_3 = tok_3 >> page_shift;
                        slot_3 = tok_3 - (page_3 << page_shift);
                        sw_3 = (grow_6 & 7) * 16;
                        base_3 = (long long)page_3 * page_stride;
                        gsrc_3 = base_3 + (long long)(slot_3 * 352);
                        dst_3 = kf4_addr + grow_6 * 128;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_3 + 36864 + (sw_3 ^ 0)), "l"(cache + (gsrc_3 + 224)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_3 + 36864 + (sw_3 ^ 16)), "l"(cache + (gsrc_3 + 224 + 16)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_3 + 36864 + (sw_3 ^ 32)), "l"(cache + (gsrc_3 + 224 + 32)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_3 + 36864 + (sw_3 ^ 48)), "l"(cache + (gsrc_3 + 224 + 48)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_3 + 36864 + (sw_3 ^ 64)), "l"(cache + (gsrc_3 + 224 + 64)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_3 + 36864 + (sw_3 ^ 80)), "l"(cache + (gsrc_3 + 224 + 80)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_3 + 36864 + (sw_3 ^ 96)), "l"(cache + (gsrc_3 + 224 + 96)));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"(dst_3 + 36864 + (sw_3 ^ 112)), "l"(cache + (gsrc_3 + 224 + 112)));
                    }
                }
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(gather_landed_p_addr + (buf_3) * 8) : "memory");
                if (load_warp_idx == 0) {
                    mbarrier_wait_hint(gather_landed_p_addr + (buf_3) * 8, it_4 >> 1 & 1, 10000000);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (it_4 >= 1) {
                        mbarrier_wait_cluster_hint(peer_free_p_addr + (buf_3) * 8, it_4 - 2 + buf_3 >> 1 & 1, 10000000);
                    }
                    if (lane == 0) {
                        uint32_t _mapa_3;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_3) : "r"(kv_landed_addr + buf_3 * 8), "r"(peer));
                        uint32_t _mapa_4;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_4) : "r"(slab_off + 36864), "r"(peer));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_4), "r"(slab_off + 36864), "r"((uint32_t)(8192)), "r"(_mapa_3)
                            : "memory");
                        mbarrier_arrive(kv_landed_addr + (buf_3) * 8);
                    }
                } else if (tile_hi_3 > t_4 + 1) {
                    int is_main_0 = 1;
                    if (t_4 + 1 >= num_main_tiles) {
                        is_main_0 = 0;
                    }
                    int tile_in_table_1 = ((is_main_0 != 0) ? t_4 + 1 : t_4 + 1 - num_main_tiles);
                    int table_width_1 = ((is_main_0 != 0) ? main_width : extra_width);
                    int* row_ptr_1 = ((is_main_0 != 0) ? (main_indices + (query_idx_3 * main_index_stride)) : (extra_indices + (query_idx_3 * extra_index_stride)));
                    int active_len_1 = table_width_1;
                    if (is_main_0 != 0) {
                        if (has_main_lengths != 0) {
                            active_len_1 = main_lengths[query_idx_3];
                        }
                    } else if (has_extra_lengths != 0) {
                        active_len_1 = extra_lengths[query_idx_3];
                    }
                    if (active_len_1 < 0) {
                        active_len_1 = 0;
                    }
                    if (active_len_1 > table_width_1) {
                        active_len_1 = table_width_1;
                    }
                    int col0_1 = tile_in_table_1 * 128 + rtid;
                    int col1_1 = col0_1 + 64;
                    int raw0_1 = -1;
                    int raw1_1 = -1;
                    if (col0_1 < table_width_1) {
                        raw0_1 = row_ptr_1[col0_1];
                    }
                    if (col1_1 < table_width_1) {
                        raw1_1 = row_ptr_1[col1_1];
                    }
                    int valid0_1 = 1;
                    if (raw0_1 < 0) {
                        valid0_1 = 0;
                    }
                    if (col0_1 >= active_len_1) {
                        valid0_1 = 0;
                    }
                    int valid1_1 = 1;
                    if (raw1_1 < 0) {
                        valid1_1 = 0;
                    }
                    if (col1_1 >= active_len_1) {
                        valid1_1 = 0;
                    }
                    int tok0_1 = ((valid0_1 != 0) ? raw0_1 : -1);
                    int tok1_1 = ((valid1_1 != 0) ? raw1_1 : -1);
                    smem_tok[(buf_3 ^ 1) * 64 + rtid] = ((rank_3 == 0) ? tok0_1 : tok1_1);
                    unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, valid0_1 != 0);
                    unsigned int bits0_1 = _vote_2;
                    unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, valid1_1 != 0);
                    unsigned int bits1_1 = _vote_3;
                    if (lane == 0) {
                        smem_mask[(it_4 + 1 & 3) * 4 + rwarp] = bits0_1;
                        smem_mask[(it_4 + 1 & 3) * 4 + 2 + rwarp] = bits1_1;
                    }
                    asm volatile("barrier.sync 10, 64;" ::: "memory");
                    mbarrier_arrive(tile_meta_addr + (it_4 + 1 & 3) * 8);
                }
            }
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
