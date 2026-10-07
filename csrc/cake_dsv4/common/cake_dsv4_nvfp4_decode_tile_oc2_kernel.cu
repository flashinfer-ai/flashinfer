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
#define TMEM_NCOLS 512
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O0_OFFSET 128
#define TMEM_TMEM_O1_OFFSET 256
#define TMEM_TMEM_O2_OFFSET 384
#define TMEM_TMEM_SFA0_OFFSET 128
#define TMEM_TMEM_SFA1_OFFSET 144
#define TMEM_TMEM_SFB0_OFFSET 160
#define TMEM_TMEM_SFB1_OFFSET 176
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
#define SMEM_SMEM_QSTAGE_OFF 54272
#define SMEM_SMEM_QSTAGE_STAGE_BYTES 16384
#define SMEM_SMEM_QSTAGE_STRIDE 16384
#define SMEM_SMEM_OSTAGE_OFF 54272
#define SMEM_SMEM_OSTAGE_STAGE_BYTES 16384
#define SMEM_SMEM_OSTAGE_STRIDE 16384
#define SMEM_SMEM_KF4_OFF 54272
#define SMEM_SMEM_KF4_STAGE_BYTES 16384
#define SMEM_SMEM_KF4_STRIDE 16384
#define SMEM_SMEM_KSF_OFF 87040
#define SMEM_SMEM_KSF_STAGE_BYTES 2048
#define SMEM_SMEM_KSF_STRIDE 2048
#define SMEM_SMEM_KSF32_OFF 87040
#define SMEM_SMEM_KSF32_STAGE_BYTES 4096
#define SMEM_SMEM_KSF32_STRIDE 4096
#define SMEM_SMEM_KROPE_OFF 91136
#define SMEM_SMEM_KROPE_STAGE_BYTES 16384
#define SMEM_SMEM_KROPE_STRIDE 16384
#define SMEM_SMEM_V_OFF 107520
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_RAW_OFF 168960
#define SMEM_SMEM_RAW_STAGE_BYTES 51200
#define SMEM_SMEM_RAW_STRIDE 51200
#define SMEM_SMEM_P_OFF 168960
#define SMEM_SMEM_P_STAGE_BYTES 16384
#define SMEM_SMEM_P_STRIDE 16384
#define SMEM_SMEM_RCPTAB_OFF 220160
#define SMEM_SMEM_RCPTAB_STAGE_BYTES 64
#define SMEM_SMEM_RCPTAB_STRIDE 64
#define SMEM_SMEM_MASK_OFF 222208
#define SMEM_SMEM_MASK_STAGE_BYTES 16
#define SMEM_SMEM_MASK_STRIDE 16
#define SMEM_SMEM_TOK_OFF 222240
#define SMEM_SMEM_TOK_STAGE_BYTES 512
#define SMEM_SMEM_TOK_STRIDE 512
#define SMEM_SMEM_ROWOFF_OFF 222752
#define SMEM_SMEM_ROWOFF_STAGE_BYTES 2048
#define SMEM_SMEM_ROWOFF_STRIDE 2048
#define SMEM_SMEM_RAW32_OFF 168960
#define SMEM_SMEM_RAW32_STAGE_BYTES 51200
#define SMEM_SMEM_RAW32_STRIDE 51200
#define SMEM_SMEM_PMAX_OFF 223264
#define SMEM_SMEM_PMAX_STAGE_BYTES 1536
#define SMEM_SMEM_PMAX_STRIDE 1536
#define SMEM_SMEM_PSUM_OFF 224800
#define SMEM_SMEM_PSUM_STAGE_BYTES 1536
#define SMEM_SMEM_PSUM_STRIDE 1536
#define SMEM_SMEM_RSUM_OFF 226336
#define SMEM_SMEM_RSUM_STAGE_BYTES 1536
#define SMEM_SMEM_RSUM_STRIDE 1536
#define SMEM_TOTAL 227968
#define THREADS 512
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsv4_nvfp4_0ccb0108d760a7cd93bc(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_out, __nv_bfloat16* __restrict__ q_rows, uint8_t* __restrict__ main_cache, uint8_t* __restrict__ extra_cache, int* __restrict__ main_indices, int* __restrict__ extra_indices, int* __restrict__ main_lengths, int* __restrict__ extra_lengths, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ lse_out, int num_heads, int num_head_tiles, int num_splits, int num_main_tiles, int main_width, int extra_width, int main_index_stride, int extra_index_stride, int has_main_lengths, int has_extra_lengths, int main_page_shift, int extra_page_shift, long long main_page_stride, long long extra_page_stride, int has_sinks, float lse_partial_scale, float lse_scale)
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
    #define kv_full_addr (mbar_base + 40)
    #define s_full_addr (mbar_base + 48)
    #define p_full_addr (mbar_base + 56)
    #define o_full_addr (mbar_base + 64)
    #define tmem_dealloc_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_qf4 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_qf4_addr = smem + 1024;
    uint8_t* smem_qsf = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_qsf_addr = smem + 33792;
    unsigned int* smem_qsf32 = reinterpret_cast<unsigned int*>(smem_raw + 33792);
    const int smem_qsf32_addr = smem + 33792;
    __nv_bfloat16* smem_qrope = reinterpret_cast<__nv_bfloat16*>(smem_raw + 37888);
    const int smem_qrope_addr = smem + 37888;
    __nv_bfloat16* smem_qstage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 54272);
    const int smem_qstage_addr = smem + 54272;
    __nv_bfloat16* smem_ostage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 54272);
    const int smem_ostage_addr = smem + 54272;
    uint8_t* smem_kf4 = reinterpret_cast<uint8_t*>(smem_raw + 54272);
    const int smem_kf4_addr = smem + 54272;
    uint8_t* smem_ksf = reinterpret_cast<uint8_t*>(smem_raw + 87040);
    const int smem_ksf_addr = smem + 87040;
    unsigned int* smem_ksf32 = reinterpret_cast<unsigned int*>(smem_raw + 87040);
    const int smem_ksf32_addr = smem + 87040;
    __nv_bfloat16* smem_krope = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
    const int smem_krope_addr = smem + 91136;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 107520);
    const int smem_v_addr = smem + 107520;
    uint8_t* smem_raw_1 = reinterpret_cast<uint8_t*>(smem_raw + 168960);
    const int smem_raw_addr = smem + 168960;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 168960);
    const int smem_p_addr = smem + 168960;
    unsigned int* smem_rcptab = reinterpret_cast<unsigned int*>(smem_raw + 220160);
    const int smem_rcptab_addr = smem + 220160;
    unsigned int* smem_mask = reinterpret_cast<unsigned int*>(smem_raw + 222208);
    const int smem_mask_addr = smem + 222208;
    int* smem_tok = reinterpret_cast<int*>(smem_raw + 222240);
    const int smem_tok_addr = smem + 222240;
    unsigned int* smem_rowoff = reinterpret_cast<unsigned int*>(smem_raw + 222752);
    const int smem_rowoff_addr = smem + 222752;
    unsigned int* smem_raw32 = reinterpret_cast<unsigned int*>(smem_raw + 168960);
    const int smem_raw32_addr = smem + 168960;
    float* smem_pmax = reinterpret_cast<float*>(smem_raw + 223264);
    const int smem_pmax_addr = smem + 223264;
    float* smem_psum = reinterpret_cast<float*>(smem_raw + 224800);
    const int smem_psum_addr = smem + 224800;
    float* smem_rsum = reinterpret_cast<float*>(smem_raw + 226336);
    const int smem_rsum_addr = smem + 226336;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_out))) : "memory");

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 11 barriers)
    // Mbarriers at smem_raw[0..88)

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
            // kv_full: 1 barriers, init_count=384
            mbarrier_init(smem + 40, 384);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // p_full: 1 barriers, init_count=384
            mbarrier_init(smem + 56, 384);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // tmem_dealloc: 1 barriers, init_count=384
            mbarrier_init(smem + 80, 384);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 88);
    if (warp == 0) {
        int _tmem_hold = smem + 88;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o0 = taddr + 128;
    const int tmem_tmem_o1 = taddr + 256;
    const int tmem_tmem_o2 = taddr + 384;
    const int tmem_tmem_sfa0 = taddr + 128;
    const int tmem_tmem_sfa1 = taddr + 144;
    const int tmem_tmem_sfb0 = taddr + 160;
    const int tmem_tmem_sfb1 = taddr + 176;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: compute0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // compute0_main
            const int local_warp = warp;
            int o_chunk = blockIdx.x % 2;
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
            int is_main = 1;
            if (split_idx >= num_main_tiles) {
                is_main = 0;
            }
            int tile_in_table = ((is_main != 0) ? split_idx : split_idx - num_main_tiles);
            int table_width = ((is_main != 0) ? main_width : extra_width);
            int* row_ptr = ((is_main != 0) ? (main_indices + (query_idx * main_index_stride)) : (extra_indices + (query_idx * extra_index_stride)));
            int col = tile_in_table * 128 + row;
            int raw_index = -1;
            if (col < table_width) {
                raw_index = row_ptr[col];
            }
            int active_len = table_width;
            if (is_main != 0) {
                if (has_main_lengths != 0) {
                    active_len = main_lengths[query_idx];
                }
            } else if (has_extra_lengths != 0) {
                active_len = extra_lengths[query_idx];
            }
            if (active_len < 0) {
                active_len = 0;
            }
            if (active_len > table_width) {
                active_len = table_width;
            }
            int valid = 1;
            if (raw_index < 0) {
                valid = 0;
            }
            if (col >= active_len) {
                valid = 0;
            }
            uint8_t* cache = ((is_main != 0) ? (main_cache) : (extra_cache));
            int page_shift = ((is_main != 0) ? main_page_shift : extra_page_shift);
            long long page_stride = ((is_main != 0) ? main_page_stride : extra_page_stride);
            int safe_index = ((raw_index >= 0) ? raw_index : 0);
            int page = safe_index >> page_shift;
            int slot_in_page = safe_index - (page << page_shift);
            int page_size = 1 << page_shift;
            long long page_base = (long long)page * page_stride;
            long long data_off = page_base + (long long)(slot_in_page * 352);
            long long sf_off = page_base + (long long)(page_size * 352 + slot_in_page * 32);
            int slot = smem_raw_addr + (unsigned int)(row * 400);
            {
                unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, valid != 0);
                unsigned int valid_bits = _vote_0;
                if (lane == 0) {
                    smem_mask[local_warp] = valid_bits;
                }
                smem_tok[row] = ((valid != 0) ? raw_index : -1);
                long long pub_d = ((valid != 0) ? data_off : (long long)-1);
                long long pub_s = ((valid != 0) ? sf_off : (long long)-1);
                unsigned int offw[4];
                offw[0] = (unsigned int)pub_d;
                offw[1] = (unsigned int)(pub_d >> 32);
                offw[2] = (unsigned int)pub_s;
                offw[3] = (unsigned int)(pub_s >> 32);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_rowoff_addr + (unsigned int)(row * 16)), "r"(*reinterpret_cast<uint32_t*>(&offw[0])), "r"(*reinterpret_cast<uint32_t*>(&offw[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&offw[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&offw[(0) + 3])));
                if (local_warp == 0) {
                    if (lane == 0) {
                        smem_rcptab[0] = 1065353216;
                        smem_rcptab[1] = 1063489081;
                        smem_rcptab[2] = 1061997773;
                        smem_rcptab[3] = 1060777612;
                        smem_rcptab[4] = 1059760811;
                        smem_rcptab[5] = 1058900441;
                        smem_rcptab[6] = 1058162981;
                        smem_rcptab[7] = 1057523849;
                        smem_rcptab[8] = 0;
                        smem_rcptab[9] = 1065353216;
                        smem_rcptab[10] = 1056964608;
                        smem_rcptab[11] = 1051372203;
                        smem_rcptab[12] = 1048576000;
                        smem_rcptab[13] = 1045220557;
                        smem_rcptab[14] = 1042983595;
                        smem_rcptab[15] = 1041385765;
                    }
                }
            }
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            const int gather_tid = warp * 32 + lane;
            const int g_chunk = gather_tid % 24;
            const int g_row0 = gather_tid / 24;
            const int g_is_sf = ((g_chunk >= 22) ? 1 : 0);
            long long g_src_off = (long long)(((g_is_sf != 0) ? 16 * (g_chunk - 14 - 8) : 16 * g_chunk));
            int g_dst0 = smem_raw_addr + (unsigned int)(g_row0 * 400) + (unsigned int)(16 * g_chunk);
            unsigned int w4[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)(g_row0 * 16)));
            long long od = (long long)w4[1] << 32 | (long long)w4[0];
            long long osf = (long long)w4[3] << 32 | (long long)w4[2];
            long long goff = ((g_is_sf != 0) ? osf : od);
            if (goff >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0), "l"(cache + (goff + g_src_off)));
            }
            unsigned int w4_0[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 16) * 16)));
            long long od_1 = (long long)w4_0[1] << 32 | (long long)w4_0[0];
            long long osf_2 = (long long)w4_0[3] << 32 | (long long)w4_0[2];
            long long goff_3 = ((g_is_sf != 0) ? osf_2 : od_1);
            if (goff_3 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 6400), "l"(cache + (goff_3 + g_src_off)));
            }
            unsigned int w4_4[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 32) * 16)));
            long long od_5 = (long long)w4_4[1] << 32 | (long long)w4_4[0];
            long long osf_6 = (long long)w4_4[3] << 32 | (long long)w4_4[2];
            long long goff_7 = ((g_is_sf != 0) ? osf_6 : od_5);
            if (goff_7 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 12800), "l"(cache + (goff_7 + g_src_off)));
            }
            unsigned int w4_8[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 48) * 16)));
            long long od_9 = (long long)w4_8[1] << 32 | (long long)w4_8[0];
            long long osf_10 = (long long)w4_8[3] << 32 | (long long)w4_8[2];
            long long goff_11 = ((g_is_sf != 0) ? osf_10 : od_9);
            if (goff_11 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 19200), "l"(cache + (goff_11 + g_src_off)));
            }
            unsigned int w4_12[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 64) * 16)));
            long long od_13 = (long long)w4_12[1] << 32 | (long long)w4_12[0];
            long long osf_14 = (long long)w4_12[3] << 32 | (long long)w4_12[2];
            long long goff_15 = ((g_is_sf != 0) ? osf_14 : od_13);
            if (goff_15 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 25600), "l"(cache + (goff_15 + g_src_off)));
            }
            unsigned int w4_16[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 80) * 16)));
            long long od_17 = (long long)w4_16[1] << 32 | (long long)w4_16[0];
            long long osf_18 = (long long)w4_16[3] << 32 | (long long)w4_16[2];
            long long goff_19 = ((g_is_sf != 0) ? osf_18 : od_17);
            if (goff_19 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 32000), "l"(cache + (goff_19 + g_src_off)));
            }
            unsigned int w4_20[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 96) * 16)));
            long long od_21 = (long long)w4_20[1] << 32 | (long long)w4_20[0];
            long long osf_22 = (long long)w4_20[3] << 32 | (long long)w4_20[2];
            long long goff_23 = ((g_is_sf != 0) ? osf_22 : od_21);
            if (goff_23 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 38400), "l"(cache + (goff_23 + g_src_off)));
            }
            unsigned int w4_24[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 112) * 16)));
            long long od_25 = (long long)w4_24[1] << 32 | (long long)w4_24[0];
            long long osf_26 = (long long)w4_24[3] << 32 | (long long)w4_24[2];
            long long goff_27 = ((g_is_sf != 0) ? osf_26 : od_25);
            if (goff_27 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 44800), "l"(cache + (goff_27 + g_src_off)));
            }
            asm volatile("cp.async.commit_group;");
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
            for (int i = 0; i < 3; i++) {
                int unit = q_warp + 12 * i;
                if (unit < 28) {
                    int q_block = unit / 7;
                    int kset = unit - q_block * 7;
                    int q_row = q_block * 32 + lane;
                    if (head_base + q_block * 32 < num_heads) {
                        int q_row_addr = smem_qstage_addr + (unsigned int)(kset * 16384) + (unsigned int)(q_row * 128);
                        unsigned int sf_word = 0;
                        for (int bp = 0; bp < 2; bp++) {
                            unsigned int words[4];
                            unsigned int qa[4];
                            unsigned int qb[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa[(0) + 3]))
                                : "r"(q_row_addr + (4 * bp ^ q_row % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb[(0) + 3]))
                                : "r"(q_row_addr + (4 * bp + 1 ^ q_row % 8) * 16));
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
                            float inv = 0.0f;
                            if (sc_exp == 0) {
                                inv = __uint_as_float(smem_rcptab[8 + sc_man]) * 512.0f;
                            } else {
                                inv = __uint_as_float(smem_rcptab[sc_man]) * __uint_as_float(134 - sc_exp << 23);
                            }
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
                            sf_word = sf_word | sc_byte << (unsigned int)(8 * (2 * bp));
                            unsigned int qa_0[4];
                            unsigned int qb_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0[(0) + 3]))
                                : "r"(q_row_addr + (4 * bp + 2 ^ q_row % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1[(0) + 3]))
                                : "r"(q_row_addr + (4 * bp + 2 + 1 ^ q_row % 8) * 16));
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
                            float inv_11 = 0.0f;
                            if (sc_exp_9 == 0) {
                                inv_11 = __uint_as_float(smem_rcptab[8 + sc_man_10]) * 512.0f;
                            } else {
                                inv_11 = __uint_as_float(smem_rcptab[sc_man_10]) * __uint_as_float(134 - sc_exp_9 << 23);
                            }
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_1 = {inv_11, inv_11};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2)[_ls], _scale2_1);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                qv_2[_ls] = qv_2[_ls] * inv_11;
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
                            sf_word = sf_word | sc_byte_8 << (unsigned int)(8 * (2 * bp + 1));
                            int chunk = 2 * kset + bp;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_qf4_addr + (unsigned int)(chunk / 8 * 16384 + (q_row * 128 + (chunk % 8 * 16 ^ q_row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words[0])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3])));
                        }
                        smem_qsf32[kset / 4 * 2048 + q_row % 32 / 8 * 512 + kset % 4 * 128 + q_row % 8 * 16 + q_row / 32 % 4 * 4 >> 2] = sf_word;
                    } else {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(2 * kset / 8 * 16384 + (q_row * 128 + (2 * kset % 8 * 16 ^ q_row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)((2 * kset + 1) / 8 * 16384 + (q_row * 128 + ((2 * kset + 1) % 8 * 16 ^ q_row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        smem_qsf32[kset / 4 * 2048 + q_row % 32 / 8 * 512 + kset % 4 * 128 + q_row % 8 * 16 + q_row / 32 % 4 * 4 >> 2] = 0;
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            unsigned int _phase_q_ready_0 = 0;
            mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0, 10000000);
            _phase_q_ready_0 ^= 1;
            asm volatile("cp.async.wait_group 0;");
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            int nope_hi_v = 8;
            int rope_pairs_hi_v = 0;
            if (valid != 0) {
                {
                    unsigned int sfw[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 3]))
                        : "r"(slot + 352));
                    smem_ksf32[row % 32 / 8 * 512 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[0];
                    smem_ksf32[row % 32 / 8 * 512 + 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[1];
                    smem_ksf32[row % 32 / 8 * 512 + 256 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[2];
                    smem_ksf32[row % 32 / 8 * 512 + 384 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[3];
                    for (int c = 0; c < nope_hi_v; c++) {
                        unsigned int raw[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 3]))
                            : "r"(slot + 16 * c));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_kf4_addr + (unsigned int)(c / 8 * 16384 + (row * 128 + (c % 8 * 16 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&raw[0])), "r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 3])));
                        unsigned int sf_word_1 = smem_raw32[row * 400 + 352 + 2 * c >> 2];
                        int block = 2 * c - 16 * o_chunk;
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
                        if (c * 2 >> 4 == o_chunk) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_v_addr + (unsigned int)(block / 8 * 16384 + (row * 128 + (block % 8 * 16 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8[0])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 3])));
                        }
                        int block_0 = 2 * c + 1 - 16 * o_chunk;
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
                        if (c * 2 >> 4 == o_chunk) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_v_addr + (unsigned int)(block_0 / 8 * 16384 + (row * 128 + (block_0 % 8 * 16 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 3])));
                        }
                    }
                }
            } else {
                {
                    for (int c_1 = 0; c_1 < nope_hi_v; c_1++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kf4_addr + (unsigned int)(c_1 / 8 * 16384 + (row * 128 + (c_1 % 8 * 16 ^ row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        if (c_1 * 2 >> 4 == o_chunk) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)((2 * c_1 - 16 * o_chunk) / 8 * 16384 + (row * 128 + ((2 * c_1 - 16 * o_chunk) % 8 * 16 ^ row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)((2 * c_1 + 1 - 16 * o_chunk) / 8 * 16384 + (row * 128 + ((2 * c_1 + 1 - 16 * o_chunk) % 8 * 16 ^ row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        }
                    }
                    smem_ksf32[row % 32 / 8 * 512 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[row % 32 / 8 * 512 + 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[row % 32 / 8 * 512 + 256 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[row % 32 / 8 * 512 + 384 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            float sink_log2 = 0.0f;
            int has_sink_row = 0;
            if (has_sinks != 0 && split_idx == 0 && row_valid != 0) {
                has_sink_row = 1;
                sink_log2 = sinks[head_row] * 1.4426950408889634f;
            }
            unsigned int _phase_s_full_0 = 0;
            mbarrier_wait_hint(s_full_addr, _phase_s_full_0, 10000000);
            _phase_s_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float slice_max = -CAKE_INF;
            float score_values[64];
            unsigned int mask_words[4];
            mask_words[0] = smem_mask[0];
            mask_words[1] = smem_mask[1];
            mask_words[2] = smem_mask[2];
            mask_words[3] = smem_mask[3];
            if (warp_rows_valid != 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(score_values[0]), "=f"(score_values[1]), "=f"(score_values[2]), "=f"(score_values[3]), "=f"(score_values[4]), "=f"(score_values[5]), "=f"(score_values[6]), "=f"(score_values[7]), "=f"(score_values[8]), "=f"(score_values[9]), "=f"(score_values[10]), "=f"(score_values[11]), "=f"(score_values[12]), "=f"(score_values[13]), "=f"(score_values[14]), "=f"(score_values[15]), "=f"(score_values[16]), "=f"(score_values[17]), "=f"(score_values[18]), "=f"(score_values[19]), "=f"(score_values[20]), "=f"(score_values[21]), "=f"(score_values[22]), "=f"(score_values[23]), "=f"(score_values[24]), "=f"(score_values[25]), "=f"(score_values[26]), "=f"(score_values[27]), "=f"(score_values[28]), "=f"(score_values[29]), "=f"(score_values[30]), "=f"(score_values[31])
                    : "r"(taddr + (unsigned int)(tmem_row_origin << 16)));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(score_values[32]), "=f"(score_values[33]), "=f"(score_values[34]), "=f"(score_values[35]), "=f"(score_values[36]), "=f"(score_values[37]), "=f"(score_values[38]), "=f"(score_values[39]), "=f"(score_values[40]), "=f"(score_values[41]), "=f"(score_values[42]), "=f"(score_values[43]), "=f"(score_values[44]), "=f"(score_values[45]), "=f"(score_values[46]), "=f"(score_values[47]), "=f"(score_values[48]), "=f"(score_values[49]), "=f"(score_values[50]), "=f"(score_values[51]), "=f"(score_values[52]), "=f"(score_values[53]), "=f"(score_values[54]), "=f"(score_values[55]), "=f"(score_values[56]), "=f"(score_values[57]), "=f"(score_values[58]), "=f"(score_values[59]), "=f"(score_values[60]), "=f"(score_values[61]), "=f"(score_values[62]), "=f"(score_values[63])
                    : "r"(taddr + (unsigned int)(tmem_row_origin << 16) + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int full_mask = mask_words[0];
                full_mask = full_mask & mask_words[1];
                if (full_mask != 4294967295u) {
                    if ((mask_words[0] & 1) == 0) {
                        score_values[0] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 1 & 1) == 0) {
                        score_values[1] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 2 & 1) == 0) {
                        score_values[2] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 3 & 1) == 0) {
                        score_values[3] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 4 & 1) == 0) {
                        score_values[4] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 5 & 1) == 0) {
                        score_values[5] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 6 & 1) == 0) {
                        score_values[6] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 7 & 1) == 0) {
                        score_values[7] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 8 & 1) == 0) {
                        score_values[8] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 9 & 1) == 0) {
                        score_values[9] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 10 & 1) == 0) {
                        score_values[10] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 11 & 1) == 0) {
                        score_values[11] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 12 & 1) == 0) {
                        score_values[12] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 13 & 1) == 0) {
                        score_values[13] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 14 & 1) == 0) {
                        score_values[14] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 15 & 1) == 0) {
                        score_values[15] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 16 & 1) == 0) {
                        score_values[16] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 17 & 1) == 0) {
                        score_values[17] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 18 & 1) == 0) {
                        score_values[18] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 19 & 1) == 0) {
                        score_values[19] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 20 & 1) == 0) {
                        score_values[20] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 21 & 1) == 0) {
                        score_values[21] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 22 & 1) == 0) {
                        score_values[22] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 23 & 1) == 0) {
                        score_values[23] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 24 & 1) == 0) {
                        score_values[24] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 25 & 1) == 0) {
                        score_values[25] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 26 & 1) == 0) {
                        score_values[26] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 27 & 1) == 0) {
                        score_values[27] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 28 & 1) == 0) {
                        score_values[28] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 29 & 1) == 0) {
                        score_values[29] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 30 & 1) == 0) {
                        score_values[30] = -CAKE_INF;
                    }
                    if ((mask_words[0] >> 31 & 1) == 0) {
                        score_values[31] = -CAKE_INF;
                    }
                    if ((mask_words[1] & 1) == 0) {
                        score_values[32] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 1 & 1) == 0) {
                        score_values[33] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 2 & 1) == 0) {
                        score_values[34] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 3 & 1) == 0) {
                        score_values[35] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 4 & 1) == 0) {
                        score_values[36] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 5 & 1) == 0) {
                        score_values[37] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 6 & 1) == 0) {
                        score_values[38] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 7 & 1) == 0) {
                        score_values[39] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 8 & 1) == 0) {
                        score_values[40] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 9 & 1) == 0) {
                        score_values[41] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 10 & 1) == 0) {
                        score_values[42] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 11 & 1) == 0) {
                        score_values[43] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 12 & 1) == 0) {
                        score_values[44] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 13 & 1) == 0) {
                        score_values[45] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 14 & 1) == 0) {
                        score_values[46] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 15 & 1) == 0) {
                        score_values[47] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 16 & 1) == 0) {
                        score_values[48] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 17 & 1) == 0) {
                        score_values[49] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 18 & 1) == 0) {
                        score_values[50] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 19 & 1) == 0) {
                        score_values[51] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 20 & 1) == 0) {
                        score_values[52] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 21 & 1) == 0) {
                        score_values[53] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 22 & 1) == 0) {
                        score_values[54] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 23 & 1) == 0) {
                        score_values[55] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 24 & 1) == 0) {
                        score_values[56] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 25 & 1) == 0) {
                        score_values[57] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 26 & 1) == 0) {
                        score_values[58] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 27 & 1) == 0) {
                        score_values[59] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 28 & 1) == 0) {
                        score_values[60] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 29 & 1) == 0) {
                        score_values[61] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 30 & 1) == 0) {
                        score_values[62] = -CAKE_INF;
                    }
                    if ((mask_words[1] >> 31 & 1) == 0) {
                        score_values[63] = -CAKE_INF;
                    }
                }
                float2 _reg_reduce_max2_10 = {-CAKE_INF, -CAKE_INF};
                row_max_x32_accum(&score_values[0], _reg_reduce_max2_10);
                row_max_x32_accum(&score_values[32], _reg_reduce_max2_10);
                float score_values_max = row_max_reduce(_reg_reduce_max2_10);
                slice_max = score_values_max;
            }
            smem_pmax[row] = slice_max;
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            float _max_30 = max_noftz(smem_pmax[row], smem_pmax[128 + row]);
            float _max_31 = max_noftz(_max_30, smem_pmax[256 + row]);
            float tile_max = _max_31;
            float row_max_scaled = tile_max * softmax_scale_log2;
            if (has_sink_row != 0) {
                float _max_32 = max_noftz(row_max_scaled, sink_log2);
                row_max_scaled = _max_32;
            }
            if (row_max_scaled == -CAKE_INF) {
                row_max_scaled = 0.0f;
            }
            float slice_sum = 0.0f;
            float rsum = 0.0f;
            if (warp_rows_valid != 0) {
                float score_bias = -row_max_scaled;
                const float2 _fma_b2_11 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_12 = {score_bias, score_bias};
                #pragma unroll
                for (int _lf = 0; _lf < 32; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>(score_values)[_lf], _fma_b2_11, _fma_c2_12);
                #pragma unroll
                for (int _le = 0; _le < 64; _le++) {
                    score_values[_le] = approx_exp2(score_values[_le]);
                }
                float2 _reg_reduce_sum2_13 = make_float2(0.0f, 0.0f);
                softmax_block_sum(&score_values[0], &_reg_reduce_sum2_13);
                softmax_block_sum(&score_values[32], &_reg_reduce_sum2_13);
                float score_values_sum = _reg_reduce_sum2_13.x + _reg_reduce_sum2_13.y;
                slice_sum = score_values_sum;
                if (row_valid == 0) {
                    score_values[0] = 0.0f;
                    score_values[1] = 0.0f;
                    score_values[2] = 0.0f;
                    score_values[3] = 0.0f;
                    score_values[4] = 0.0f;
                    score_values[5] = 0.0f;
                    score_values[6] = 0.0f;
                    score_values[7] = 0.0f;
                    score_values[8] = 0.0f;
                    score_values[9] = 0.0f;
                    score_values[10] = 0.0f;
                    score_values[11] = 0.0f;
                    score_values[12] = 0.0f;
                    score_values[13] = 0.0f;
                    score_values[14] = 0.0f;
                    score_values[15] = 0.0f;
                    score_values[16] = 0.0f;
                    score_values[17] = 0.0f;
                    score_values[18] = 0.0f;
                    score_values[19] = 0.0f;
                    score_values[20] = 0.0f;
                    score_values[21] = 0.0f;
                    score_values[22] = 0.0f;
                    score_values[23] = 0.0f;
                    score_values[24] = 0.0f;
                    score_values[25] = 0.0f;
                    score_values[26] = 0.0f;
                    score_values[27] = 0.0f;
                    score_values[28] = 0.0f;
                    score_values[29] = 0.0f;
                    score_values[30] = 0.0f;
                    score_values[31] = 0.0f;
                    score_values[32] = 0.0f;
                    score_values[33] = 0.0f;
                    score_values[34] = 0.0f;
                    score_values[35] = 0.0f;
                    score_values[36] = 0.0f;
                    score_values[37] = 0.0f;
                    score_values[38] = 0.0f;
                    score_values[39] = 0.0f;
                    score_values[40] = 0.0f;
                    score_values[41] = 0.0f;
                    score_values[42] = 0.0f;
                    score_values[43] = 0.0f;
                    score_values[44] = 0.0f;
                    score_values[45] = 0.0f;
                    score_values[46] = 0.0f;
                    score_values[47] = 0.0f;
                    score_values[48] = 0.0f;
                    score_values[49] = 0.0f;
                    score_values[50] = 0.0f;
                    score_values[51] = 0.0f;
                    score_values[52] = 0.0f;
                    score_values[53] = 0.0f;
                    score_values[54] = 0.0f;
                    score_values[55] = 0.0f;
                    score_values[56] = 0.0f;
                    score_values[57] = 0.0f;
                    score_values[58] = 0.0f;
                    score_values[59] = 0.0f;
                    score_values[60] = 0.0f;
                    score_values[61] = 0.0f;
                    score_values[62] = 0.0f;
                    score_values[63] = 0.0f;
                }
                unsigned int packed_p[16];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(score_values[0]), "f"(score_values[1]),
                                           "f"(score_values[2]), "f"(score_values[3]));
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
                        : "=r"(_packed) : "f"(score_values[4]), "f"(score_values[5]),
                                           "f"(score_values[6]), "f"(score_values[7]));
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
                        : "=r"(_packed) : "f"(score_values[8]), "f"(score_values[9]),
                                           "f"(score_values[10]), "f"(score_values[11]));
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
                        : "=r"(_packed) : "f"(score_values[12]), "f"(score_values[13]),
                                           "f"(score_values[14]), "f"(score_values[15]));
                    packed_p[3] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[16]), "f"(score_values[17]),
                                           "f"(score_values[18]), "f"(score_values[19]));
                    packed_p[4] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[20]), "f"(score_values[21]),
                                           "f"(score_values[22]), "f"(score_values[23]));
                    packed_p[5] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[24]), "f"(score_values[25]),
                                           "f"(score_values[26]), "f"(score_values[27]));
                    packed_p[6] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[28]), "f"(score_values[29]),
                                           "f"(score_values[30]), "f"(score_values[31]));
                    packed_p[7] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[32]), "f"(score_values[33]),
                                           "f"(score_values[34]), "f"(score_values[35]));
                    packed_p[8] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[36]), "f"(score_values[37]),
                                           "f"(score_values[38]), "f"(score_values[39]));
                    packed_p[9] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[40]), "f"(score_values[41]),
                                           "f"(score_values[42]), "f"(score_values[43]));
                    packed_p[10] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[44]), "f"(score_values[45]),
                                           "f"(score_values[46]), "f"(score_values[47]));
                    packed_p[11] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[48]), "f"(score_values[49]),
                                           "f"(score_values[50]), "f"(score_values[51]));
                    packed_p[12] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[52]), "f"(score_values[53]),
                                           "f"(score_values[54]), "f"(score_values[55]));
                    packed_p[13] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[56]), "f"(score_values[57]),
                                           "f"(score_values[58]), "f"(score_values[59]));
                    packed_p[14] = _packed;
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
                        : "=r"(_packed) : "f"(score_values[60]), "f"(score_values[61]),
                                           "f"(score_values[62]), "f"(score_values[63]));
                    packed_p[15] = _packed;
                }
                float _fp8_rt_0;
                uint16_t _e4m3x2_14;
                uint32_t _f16x2_14;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_14) : "f"(0.0f), "f"(score_values[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_14) : "h"(_e4m3x2_14));
                uint16_t _fp8_h0_14 = (uint16_t)(_f16x2_14 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_14));
                rsum = rsum + _fp8_rt_0;
                float _fp8_rt_1;
                uint16_t _e4m3x2_15;
                uint32_t _f16x2_15;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_15) : "f"(0.0f), "f"(score_values[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_15) : "h"(_e4m3x2_15));
                uint16_t _fp8_h0_15 = (uint16_t)(_f16x2_15 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_15));
                rsum = rsum + _fp8_rt_1;
                float _fp8_rt_2;
                uint16_t _e4m3x2_16;
                uint32_t _f16x2_16;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_16) : "f"(0.0f), "f"(score_values[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_16) : "h"(_e4m3x2_16));
                uint16_t _fp8_h0_16 = (uint16_t)(_f16x2_16 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_2) : "h"(_fp8_h0_16));
                rsum = rsum + _fp8_rt_2;
                float _fp8_rt_3;
                uint16_t _e4m3x2_17;
                uint32_t _f16x2_17;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_17) : "f"(0.0f), "f"(score_values[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_17) : "h"(_e4m3x2_17));
                uint16_t _fp8_h0_17 = (uint16_t)(_f16x2_17 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_3) : "h"(_fp8_h0_17));
                rsum = rsum + _fp8_rt_3;
                float _fp8_rt_4;
                uint16_t _e4m3x2_18;
                uint32_t _f16x2_18;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_18) : "f"(0.0f), "f"(score_values[4]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_18) : "h"(_e4m3x2_18));
                uint16_t _fp8_h0_18 = (uint16_t)(_f16x2_18 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_4) : "h"(_fp8_h0_18));
                rsum = rsum + _fp8_rt_4;
                float _fp8_rt_5;
                uint16_t _e4m3x2_19;
                uint32_t _f16x2_19;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_19) : "f"(0.0f), "f"(score_values[5]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_19) : "h"(_e4m3x2_19));
                uint16_t _fp8_h0_19 = (uint16_t)(_f16x2_19 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_5) : "h"(_fp8_h0_19));
                rsum = rsum + _fp8_rt_5;
                float _fp8_rt_6;
                uint16_t _e4m3x2_20;
                uint32_t _f16x2_20;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_20) : "f"(0.0f), "f"(score_values[6]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_20) : "h"(_e4m3x2_20));
                uint16_t _fp8_h0_20 = (uint16_t)(_f16x2_20 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_6) : "h"(_fp8_h0_20));
                rsum = rsum + _fp8_rt_6;
                float _fp8_rt_7;
                uint16_t _e4m3x2_21;
                uint32_t _f16x2_21;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_21) : "f"(0.0f), "f"(score_values[7]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_21) : "h"(_e4m3x2_21));
                uint16_t _fp8_h0_21 = (uint16_t)(_f16x2_21 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_7) : "h"(_fp8_h0_21));
                rsum = rsum + _fp8_rt_7;
                float _fp8_rt_8;
                uint16_t _e4m3x2_22;
                uint32_t _f16x2_22;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_22) : "f"(0.0f), "f"(score_values[8]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_22) : "h"(_e4m3x2_22));
                uint16_t _fp8_h0_22 = (uint16_t)(_f16x2_22 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_8) : "h"(_fp8_h0_22));
                rsum = rsum + _fp8_rt_8;
                float _fp8_rt_9;
                uint16_t _e4m3x2_23;
                uint32_t _f16x2_23;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_23) : "f"(0.0f), "f"(score_values[9]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_23) : "h"(_e4m3x2_23));
                uint16_t _fp8_h0_23 = (uint16_t)(_f16x2_23 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_9) : "h"(_fp8_h0_23));
                rsum = rsum + _fp8_rt_9;
                float _fp8_rt_10;
                uint16_t _e4m3x2_24;
                uint32_t _f16x2_24;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_24) : "f"(0.0f), "f"(score_values[10]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_24) : "h"(_e4m3x2_24));
                uint16_t _fp8_h0_24 = (uint16_t)(_f16x2_24 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_10) : "h"(_fp8_h0_24));
                rsum = rsum + _fp8_rt_10;
                float _fp8_rt_11;
                uint16_t _e4m3x2_25;
                uint32_t _f16x2_25;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_25) : "f"(0.0f), "f"(score_values[11]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_25) : "h"(_e4m3x2_25));
                uint16_t _fp8_h0_25 = (uint16_t)(_f16x2_25 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_11) : "h"(_fp8_h0_25));
                rsum = rsum + _fp8_rt_11;
                float _fp8_rt_12;
                uint16_t _e4m3x2_26;
                uint32_t _f16x2_26;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_26) : "f"(0.0f), "f"(score_values[12]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_26) : "h"(_e4m3x2_26));
                uint16_t _fp8_h0_26 = (uint16_t)(_f16x2_26 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_12) : "h"(_fp8_h0_26));
                rsum = rsum + _fp8_rt_12;
                float _fp8_rt_13;
                uint16_t _e4m3x2_27;
                uint32_t _f16x2_27;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_27) : "f"(0.0f), "f"(score_values[13]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_27) : "h"(_e4m3x2_27));
                uint16_t _fp8_h0_27 = (uint16_t)(_f16x2_27 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_13) : "h"(_fp8_h0_27));
                rsum = rsum + _fp8_rt_13;
                float _fp8_rt_14;
                uint16_t _e4m3x2_28;
                uint32_t _f16x2_28;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_28) : "f"(0.0f), "f"(score_values[14]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_28) : "h"(_e4m3x2_28));
                uint16_t _fp8_h0_28 = (uint16_t)(_f16x2_28 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_14) : "h"(_fp8_h0_28));
                rsum = rsum + _fp8_rt_14;
                float _fp8_rt_15;
                uint16_t _e4m3x2_29;
                uint32_t _f16x2_29;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_29) : "f"(0.0f), "f"(score_values[15]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_29) : "h"(_e4m3x2_29));
                uint16_t _fp8_h0_29 = (uint16_t)(_f16x2_29 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_15) : "h"(_fp8_h0_29));
                rsum = rsum + _fp8_rt_15;
                float _fp8_rt_16;
                uint16_t _e4m3x2_30;
                uint32_t _f16x2_30;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_30) : "f"(0.0f), "f"(score_values[16]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_30) : "h"(_e4m3x2_30));
                uint16_t _fp8_h0_30 = (uint16_t)(_f16x2_30 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_16) : "h"(_fp8_h0_30));
                rsum = rsum + _fp8_rt_16;
                float _fp8_rt_17;
                uint16_t _e4m3x2_31;
                uint32_t _f16x2_31;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_31) : "f"(0.0f), "f"(score_values[17]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_31) : "h"(_e4m3x2_31));
                uint16_t _fp8_h0_31 = (uint16_t)(_f16x2_31 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_17) : "h"(_fp8_h0_31));
                rsum = rsum + _fp8_rt_17;
                float _fp8_rt_18;
                uint16_t _e4m3x2_32;
                uint32_t _f16x2_32;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_32) : "f"(0.0f), "f"(score_values[18]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_32) : "h"(_e4m3x2_32));
                uint16_t _fp8_h0_32 = (uint16_t)(_f16x2_32 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_18) : "h"(_fp8_h0_32));
                rsum = rsum + _fp8_rt_18;
                float _fp8_rt_19;
                uint16_t _e4m3x2_33;
                uint32_t _f16x2_33;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_33) : "f"(0.0f), "f"(score_values[19]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_33) : "h"(_e4m3x2_33));
                uint16_t _fp8_h0_33 = (uint16_t)(_f16x2_33 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_19) : "h"(_fp8_h0_33));
                rsum = rsum + _fp8_rt_19;
                float _fp8_rt_20;
                uint16_t _e4m3x2_34;
                uint32_t _f16x2_34;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_34) : "f"(0.0f), "f"(score_values[20]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_34) : "h"(_e4m3x2_34));
                uint16_t _fp8_h0_34 = (uint16_t)(_f16x2_34 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_20) : "h"(_fp8_h0_34));
                rsum = rsum + _fp8_rt_20;
                float _fp8_rt_21;
                uint16_t _e4m3x2_35;
                uint32_t _f16x2_35;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_35) : "f"(0.0f), "f"(score_values[21]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_35) : "h"(_e4m3x2_35));
                uint16_t _fp8_h0_35 = (uint16_t)(_f16x2_35 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_21) : "h"(_fp8_h0_35));
                rsum = rsum + _fp8_rt_21;
                float _fp8_rt_22;
                uint16_t _e4m3x2_36;
                uint32_t _f16x2_36;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_36) : "f"(0.0f), "f"(score_values[22]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_36) : "h"(_e4m3x2_36));
                uint16_t _fp8_h0_36 = (uint16_t)(_f16x2_36 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_22) : "h"(_fp8_h0_36));
                rsum = rsum + _fp8_rt_22;
                float _fp8_rt_23;
                uint16_t _e4m3x2_37;
                uint32_t _f16x2_37;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_37) : "f"(0.0f), "f"(score_values[23]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_37) : "h"(_e4m3x2_37));
                uint16_t _fp8_h0_37 = (uint16_t)(_f16x2_37 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_23) : "h"(_fp8_h0_37));
                rsum = rsum + _fp8_rt_23;
                float _fp8_rt_24;
                uint16_t _e4m3x2_38;
                uint32_t _f16x2_38;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_38) : "f"(0.0f), "f"(score_values[24]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_38) : "h"(_e4m3x2_38));
                uint16_t _fp8_h0_38 = (uint16_t)(_f16x2_38 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_24) : "h"(_fp8_h0_38));
                rsum = rsum + _fp8_rt_24;
                float _fp8_rt_25;
                uint16_t _e4m3x2_39;
                uint32_t _f16x2_39;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_39) : "f"(0.0f), "f"(score_values[25]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_39) : "h"(_e4m3x2_39));
                uint16_t _fp8_h0_39 = (uint16_t)(_f16x2_39 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_25) : "h"(_fp8_h0_39));
                rsum = rsum + _fp8_rt_25;
                float _fp8_rt_26;
                uint16_t _e4m3x2_40;
                uint32_t _f16x2_40;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_40) : "f"(0.0f), "f"(score_values[26]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_40) : "h"(_e4m3x2_40));
                uint16_t _fp8_h0_40 = (uint16_t)(_f16x2_40 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_26) : "h"(_fp8_h0_40));
                rsum = rsum + _fp8_rt_26;
                float _fp8_rt_27;
                uint16_t _e4m3x2_41;
                uint32_t _f16x2_41;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_41) : "f"(0.0f), "f"(score_values[27]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_41) : "h"(_e4m3x2_41));
                uint16_t _fp8_h0_41 = (uint16_t)(_f16x2_41 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_27) : "h"(_fp8_h0_41));
                rsum = rsum + _fp8_rt_27;
                float _fp8_rt_28;
                uint16_t _e4m3x2_42;
                uint32_t _f16x2_42;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_42) : "f"(0.0f), "f"(score_values[28]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_42) : "h"(_e4m3x2_42));
                uint16_t _fp8_h0_42 = (uint16_t)(_f16x2_42 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_28) : "h"(_fp8_h0_42));
                rsum = rsum + _fp8_rt_28;
                float _fp8_rt_29;
                uint16_t _e4m3x2_43;
                uint32_t _f16x2_43;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_43) : "f"(0.0f), "f"(score_values[29]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_43) : "h"(_e4m3x2_43));
                uint16_t _fp8_h0_43 = (uint16_t)(_f16x2_43 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_29) : "h"(_fp8_h0_43));
                rsum = rsum + _fp8_rt_29;
                float _fp8_rt_30;
                uint16_t _e4m3x2_44;
                uint32_t _f16x2_44;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_44) : "f"(0.0f), "f"(score_values[30]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_44) : "h"(_e4m3x2_44));
                uint16_t _fp8_h0_44 = (uint16_t)(_f16x2_44 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_30) : "h"(_fp8_h0_44));
                rsum = rsum + _fp8_rt_30;
                float _fp8_rt_31;
                uint16_t _e4m3x2_45;
                uint32_t _f16x2_45;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_45) : "f"(0.0f), "f"(score_values[31]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_45) : "h"(_e4m3x2_45));
                uint16_t _fp8_h0_45 = (uint16_t)(_f16x2_45 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_31) : "h"(_fp8_h0_45));
                rsum = rsum + _fp8_rt_31;
                float _fp8_rt_32;
                uint16_t _e4m3x2_46;
                uint32_t _f16x2_46;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_46) : "f"(0.0f), "f"(score_values[32]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_46) : "h"(_e4m3x2_46));
                uint16_t _fp8_h0_46 = (uint16_t)(_f16x2_46 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_32) : "h"(_fp8_h0_46));
                rsum = rsum + _fp8_rt_32;
                float _fp8_rt_33;
                uint16_t _e4m3x2_47;
                uint32_t _f16x2_47;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_47) : "f"(0.0f), "f"(score_values[33]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_47) : "h"(_e4m3x2_47));
                uint16_t _fp8_h0_47 = (uint16_t)(_f16x2_47 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_33) : "h"(_fp8_h0_47));
                rsum = rsum + _fp8_rt_33;
                float _fp8_rt_34;
                uint16_t _e4m3x2_48;
                uint32_t _f16x2_48;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_48) : "f"(0.0f), "f"(score_values[34]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_48) : "h"(_e4m3x2_48));
                uint16_t _fp8_h0_48 = (uint16_t)(_f16x2_48 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_34) : "h"(_fp8_h0_48));
                rsum = rsum + _fp8_rt_34;
                float _fp8_rt_35;
                uint16_t _e4m3x2_49;
                uint32_t _f16x2_49;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_49) : "f"(0.0f), "f"(score_values[35]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_49) : "h"(_e4m3x2_49));
                uint16_t _fp8_h0_49 = (uint16_t)(_f16x2_49 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_35) : "h"(_fp8_h0_49));
                rsum = rsum + _fp8_rt_35;
                float _fp8_rt_36;
                uint16_t _e4m3x2_50;
                uint32_t _f16x2_50;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_50) : "f"(0.0f), "f"(score_values[36]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_50) : "h"(_e4m3x2_50));
                uint16_t _fp8_h0_50 = (uint16_t)(_f16x2_50 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_36) : "h"(_fp8_h0_50));
                rsum = rsum + _fp8_rt_36;
                float _fp8_rt_37;
                uint16_t _e4m3x2_51;
                uint32_t _f16x2_51;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_51) : "f"(0.0f), "f"(score_values[37]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_51) : "h"(_e4m3x2_51));
                uint16_t _fp8_h0_51 = (uint16_t)(_f16x2_51 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_37) : "h"(_fp8_h0_51));
                rsum = rsum + _fp8_rt_37;
                float _fp8_rt_38;
                uint16_t _e4m3x2_52;
                uint32_t _f16x2_52;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_52) : "f"(0.0f), "f"(score_values[38]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_52) : "h"(_e4m3x2_52));
                uint16_t _fp8_h0_52 = (uint16_t)(_f16x2_52 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_38) : "h"(_fp8_h0_52));
                rsum = rsum + _fp8_rt_38;
                float _fp8_rt_39;
                uint16_t _e4m3x2_53;
                uint32_t _f16x2_53;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_53) : "f"(0.0f), "f"(score_values[39]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_53) : "h"(_e4m3x2_53));
                uint16_t _fp8_h0_53 = (uint16_t)(_f16x2_53 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_39) : "h"(_fp8_h0_53));
                rsum = rsum + _fp8_rt_39;
                float _fp8_rt_40;
                uint16_t _e4m3x2_54;
                uint32_t _f16x2_54;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_54) : "f"(0.0f), "f"(score_values[40]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_54) : "h"(_e4m3x2_54));
                uint16_t _fp8_h0_54 = (uint16_t)(_f16x2_54 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_40) : "h"(_fp8_h0_54));
                rsum = rsum + _fp8_rt_40;
                float _fp8_rt_41;
                uint16_t _e4m3x2_55;
                uint32_t _f16x2_55;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_55) : "f"(0.0f), "f"(score_values[41]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_55) : "h"(_e4m3x2_55));
                uint16_t _fp8_h0_55 = (uint16_t)(_f16x2_55 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_41) : "h"(_fp8_h0_55));
                rsum = rsum + _fp8_rt_41;
                float _fp8_rt_42;
                uint16_t _e4m3x2_56;
                uint32_t _f16x2_56;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_56) : "f"(0.0f), "f"(score_values[42]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_56) : "h"(_e4m3x2_56));
                uint16_t _fp8_h0_56 = (uint16_t)(_f16x2_56 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_42) : "h"(_fp8_h0_56));
                rsum = rsum + _fp8_rt_42;
                float _fp8_rt_43;
                uint16_t _e4m3x2_57;
                uint32_t _f16x2_57;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_57) : "f"(0.0f), "f"(score_values[43]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_57) : "h"(_e4m3x2_57));
                uint16_t _fp8_h0_57 = (uint16_t)(_f16x2_57 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_43) : "h"(_fp8_h0_57));
                rsum = rsum + _fp8_rt_43;
                float _fp8_rt_44;
                uint16_t _e4m3x2_58;
                uint32_t _f16x2_58;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_58) : "f"(0.0f), "f"(score_values[44]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_58) : "h"(_e4m3x2_58));
                uint16_t _fp8_h0_58 = (uint16_t)(_f16x2_58 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_44) : "h"(_fp8_h0_58));
                rsum = rsum + _fp8_rt_44;
                float _fp8_rt_45;
                uint16_t _e4m3x2_59;
                uint32_t _f16x2_59;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_59) : "f"(0.0f), "f"(score_values[45]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_59) : "h"(_e4m3x2_59));
                uint16_t _fp8_h0_59 = (uint16_t)(_f16x2_59 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_45) : "h"(_fp8_h0_59));
                rsum = rsum + _fp8_rt_45;
                float _fp8_rt_46;
                uint16_t _e4m3x2_60;
                uint32_t _f16x2_60;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_60) : "f"(0.0f), "f"(score_values[46]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_60) : "h"(_e4m3x2_60));
                uint16_t _fp8_h0_60 = (uint16_t)(_f16x2_60 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_46) : "h"(_fp8_h0_60));
                rsum = rsum + _fp8_rt_46;
                float _fp8_rt_47;
                uint16_t _e4m3x2_61;
                uint32_t _f16x2_61;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_61) : "f"(0.0f), "f"(score_values[47]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_61) : "h"(_e4m3x2_61));
                uint16_t _fp8_h0_61 = (uint16_t)(_f16x2_61 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_47) : "h"(_fp8_h0_61));
                rsum = rsum + _fp8_rt_47;
                float _fp8_rt_48;
                uint16_t _e4m3x2_62;
                uint32_t _f16x2_62;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_62) : "f"(0.0f), "f"(score_values[48]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_62) : "h"(_e4m3x2_62));
                uint16_t _fp8_h0_62 = (uint16_t)(_f16x2_62 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_48) : "h"(_fp8_h0_62));
                rsum = rsum + _fp8_rt_48;
                float _fp8_rt_49;
                uint16_t _e4m3x2_63;
                uint32_t _f16x2_63;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_63) : "f"(0.0f), "f"(score_values[49]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_63) : "h"(_e4m3x2_63));
                uint16_t _fp8_h0_63 = (uint16_t)(_f16x2_63 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_49) : "h"(_fp8_h0_63));
                rsum = rsum + _fp8_rt_49;
                float _fp8_rt_50;
                uint16_t _e4m3x2_64;
                uint32_t _f16x2_64;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_64) : "f"(0.0f), "f"(score_values[50]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_64) : "h"(_e4m3x2_64));
                uint16_t _fp8_h0_64 = (uint16_t)(_f16x2_64 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_50) : "h"(_fp8_h0_64));
                rsum = rsum + _fp8_rt_50;
                float _fp8_rt_51;
                uint16_t _e4m3x2_65;
                uint32_t _f16x2_65;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_65) : "f"(0.0f), "f"(score_values[51]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_65) : "h"(_e4m3x2_65));
                uint16_t _fp8_h0_65 = (uint16_t)(_f16x2_65 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_51) : "h"(_fp8_h0_65));
                rsum = rsum + _fp8_rt_51;
                float _fp8_rt_52;
                uint16_t _e4m3x2_66;
                uint32_t _f16x2_66;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_66) : "f"(0.0f), "f"(score_values[52]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_66) : "h"(_e4m3x2_66));
                uint16_t _fp8_h0_66 = (uint16_t)(_f16x2_66 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_52) : "h"(_fp8_h0_66));
                rsum = rsum + _fp8_rt_52;
                float _fp8_rt_53;
                uint16_t _e4m3x2_67;
                uint32_t _f16x2_67;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_67) : "f"(0.0f), "f"(score_values[53]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_67) : "h"(_e4m3x2_67));
                uint16_t _fp8_h0_67 = (uint16_t)(_f16x2_67 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_53) : "h"(_fp8_h0_67));
                rsum = rsum + _fp8_rt_53;
                float _fp8_rt_54;
                uint16_t _e4m3x2_68;
                uint32_t _f16x2_68;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_68) : "f"(0.0f), "f"(score_values[54]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_68) : "h"(_e4m3x2_68));
                uint16_t _fp8_h0_68 = (uint16_t)(_f16x2_68 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_54) : "h"(_fp8_h0_68));
                rsum = rsum + _fp8_rt_54;
                float _fp8_rt_55;
                uint16_t _e4m3x2_69;
                uint32_t _f16x2_69;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_69) : "f"(0.0f), "f"(score_values[55]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_69) : "h"(_e4m3x2_69));
                uint16_t _fp8_h0_69 = (uint16_t)(_f16x2_69 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_55) : "h"(_fp8_h0_69));
                rsum = rsum + _fp8_rt_55;
                float _fp8_rt_56;
                uint16_t _e4m3x2_70;
                uint32_t _f16x2_70;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_70) : "f"(0.0f), "f"(score_values[56]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_70) : "h"(_e4m3x2_70));
                uint16_t _fp8_h0_70 = (uint16_t)(_f16x2_70 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_56) : "h"(_fp8_h0_70));
                rsum = rsum + _fp8_rt_56;
                float _fp8_rt_57;
                uint16_t _e4m3x2_71;
                uint32_t _f16x2_71;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_71) : "f"(0.0f), "f"(score_values[57]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_71) : "h"(_e4m3x2_71));
                uint16_t _fp8_h0_71 = (uint16_t)(_f16x2_71 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_57) : "h"(_fp8_h0_71));
                rsum = rsum + _fp8_rt_57;
                float _fp8_rt_58;
                uint16_t _e4m3x2_72;
                uint32_t _f16x2_72;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_72) : "f"(0.0f), "f"(score_values[58]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_72) : "h"(_e4m3x2_72));
                uint16_t _fp8_h0_72 = (uint16_t)(_f16x2_72 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_58) : "h"(_fp8_h0_72));
                rsum = rsum + _fp8_rt_58;
                float _fp8_rt_59;
                uint16_t _e4m3x2_73;
                uint32_t _f16x2_73;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_73) : "f"(0.0f), "f"(score_values[59]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_73) : "h"(_e4m3x2_73));
                uint16_t _fp8_h0_73 = (uint16_t)(_f16x2_73 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_59) : "h"(_fp8_h0_73));
                rsum = rsum + _fp8_rt_59;
                float _fp8_rt_60;
                uint16_t _e4m3x2_74;
                uint32_t _f16x2_74;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_74) : "f"(0.0f), "f"(score_values[60]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_74) : "h"(_e4m3x2_74));
                uint16_t _fp8_h0_74 = (uint16_t)(_f16x2_74 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_60) : "h"(_fp8_h0_74));
                rsum = rsum + _fp8_rt_60;
                float _fp8_rt_61;
                uint16_t _e4m3x2_75;
                uint32_t _f16x2_75;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_75) : "f"(0.0f), "f"(score_values[61]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_75) : "h"(_e4m3x2_75));
                uint16_t _fp8_h0_75 = (uint16_t)(_f16x2_75 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_61) : "h"(_fp8_h0_75));
                rsum = rsum + _fp8_rt_61;
                float _fp8_rt_62;
                uint16_t _e4m3x2_76;
                uint32_t _f16x2_76;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_76) : "f"(0.0f), "f"(score_values[62]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_76) : "h"(_e4m3x2_76));
                uint16_t _fp8_h0_76 = (uint16_t)(_f16x2_76 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_62) : "h"(_fp8_h0_76));
                rsum = rsum + _fp8_rt_62;
                float _fp8_rt_63;
                uint16_t _e4m3x2_77;
                uint32_t _f16x2_77;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_77) : "f"(0.0f), "f"(score_values[63]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_77) : "h"(_e4m3x2_77));
                uint16_t _fp8_h0_77 = (uint16_t)(_f16x2_77 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_63) : "h"(_fp8_h0_77));
                rsum = rsum + _fp8_rt_63;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_p_addr + (unsigned int)(row * 128 + (0 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_p_addr + (unsigned int)(row * 128 + (16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(4) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_p_addr + (unsigned int)(row * 128 + (32 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p[8])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(8) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_p_addr + (unsigned int)(row * 128 + (48 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p[12])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p[(12) + 3])));
            }
            float sink_term = 0.0f;
            if (has_sink_row != 0) {
                float _exp2_0 = approx_exp2(sink_log2 - row_max_scaled);
                sink_term = _exp2_0;
            }
            smem_psum[row] = slice_sum;
            smem_rsum[row] = rsum;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(p_full_addr);
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            if (row_valid != 0 && o_chunk == 0) {
                float row_sum = smem_psum[row] + smem_psum[128 + row] + smem_psum[256 + row] + sink_term;
                int lse_offset = (query_idx * num_heads + head_row) * num_splits + split_idx;
                float _log2_0;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(row_sum));
                partial_lse[lse_offset] = ((row_sum > 0.0f) ? (row_max_scaled + _log2_0) * lse_partial_scale : -CAKE_INF);
            }
            float denom = smem_rsum[row] + smem_rsum[128 + row] + smem_rsum[256 + row] + sink_term;
            float _rcp_0 = approx_rcp(denom);
            float norm = ((denom > 0.0f) ? _rcp_0 * output_scale : 0.0f);
            float o_values[64];
            unsigned int packed[32];
            long long out_base = ((long long)(query_idx * num_heads + head_row) * (long long)num_splits + (long long)split_idx) * 512;
            unsigned int _phase_o_full_0 = 0;
            if (num_heads <= 32) {
                mbarrier_wait_hint(o_full_addr, _phase_o_full_0, 10000000);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp_rows_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values[0]), "=f"(o_values[1]), "=f"(o_values[2]), "=f"(o_values[3]), "=f"(o_values[4]), "=f"(o_values[5]), "=f"(o_values[6]), "=f"(o_values[7]), "=f"(o_values[8]), "=f"(o_values[9]), "=f"(o_values[10]), "=f"(o_values[11]), "=f"(o_values[12]), "=f"(o_values[13]), "=f"(o_values[14]), "=f"(o_values[15]), "=f"(o_values[16]), "=f"(o_values[17]), "=f"(o_values[18]), "=f"(o_values[19]), "=f"(o_values[20]), "=f"(o_values[21]), "=f"(o_values[22]), "=f"(o_values[23]), "=f"(o_values[24]), "=f"(o_values[25]), "=f"(o_values[26]), "=f"(o_values[27]), "=f"(o_values[28]), "=f"(o_values[29]), "=f"(o_values[30]), "=f"(o_values[31])
                        : "r"(taddr + 128 + (unsigned int)(tmem_row_origin << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values[32]), "=f"(o_values[33]), "=f"(o_values[34]), "=f"(o_values[35]), "=f"(o_values[36]), "=f"(o_values[37]), "=f"(o_values[38]), "=f"(o_values[39]), "=f"(o_values[40]), "=f"(o_values[41]), "=f"(o_values[42]), "=f"(o_values[43]), "=f"(o_values[44]), "=f"(o_values[45]), "=f"(o_values[46]), "=f"(o_values[47]), "=f"(o_values[48]), "=f"(o_values[49]), "=f"(o_values[50]), "=f"(o_values[51]), "=f"(o_values[52]), "=f"(o_values[53]), "=f"(o_values[54]), "=f"(o_values[55]), "=f"(o_values[56]), "=f"(o_values[57]), "=f"(o_values[58]), "=f"(o_values[59]), "=f"(o_values[60]), "=f"(o_values[61]), "=f"(o_values[62]), "=f"(o_values[63])
                        : "r"(taddr + 128 + (unsigned int)(tmem_row_origin << 16) + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_78 = {norm, norm};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values)[_ls], _scale2_78);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++) {
                        o_values[_ls] = o_values[_ls] * norm;
                    }
                    #endif
                    if (row_valid != 0) {
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values[0 + 0], o_values[0 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values[0 + 2], o_values[0 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values[0 + 4], o_values[0 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values[0 + 6], o_values[0 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values[0 + 8], o_values[0 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values[0 + 10], o_values[0 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values[0 + 12], o_values[0 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values[0 + 14], o_values[0 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base + (long long)(o_chunk * 2 * 128))))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values[16 + 0], o_values[16 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values[16 + 2], o_values[16 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values[16 + 4], o_values[16 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values[16 + 6], o_values[16 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values[16 + 8], o_values[16 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values[16 + 10], o_values[16 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values[16 + 12], o_values[16 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values[16 + 14], o_values[16 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base + (long long)(o_chunk * 2 * 128) + 16)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values[32 + 0], o_values[32 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values[32 + 2], o_values[32 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values[32 + 4], o_values[32 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values[32 + 6], o_values[32 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values[32 + 8], o_values[32 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values[32 + 10], o_values[32 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values[32 + 12], o_values[32 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values[32 + 14], o_values[32 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base + (long long)(o_chunk * 2 * 128) + 32)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values[48 + 0], o_values[48 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values[48 + 2], o_values[48 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values[48 + 4], o_values[48 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values[48 + 6], o_values[48 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values[48 + 8], o_values[48 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values[48 + 10], o_values[48 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values[48 + 12], o_values[48 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values[48 + 14], o_values[48 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base + (long long)(o_chunk * 2 * 128) + 48)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values[0]), "=f"(o_values[1]), "=f"(o_values[2]), "=f"(o_values[3]), "=f"(o_values[4]), "=f"(o_values[5]), "=f"(o_values[6]), "=f"(o_values[7]), "=f"(o_values[8]), "=f"(o_values[9]), "=f"(o_values[10]), "=f"(o_values[11]), "=f"(o_values[12]), "=f"(o_values[13]), "=f"(o_values[14]), "=f"(o_values[15]), "=f"(o_values[16]), "=f"(o_values[17]), "=f"(o_values[18]), "=f"(o_values[19]), "=f"(o_values[20]), "=f"(o_values[21]), "=f"(o_values[22]), "=f"(o_values[23]), "=f"(o_values[24]), "=f"(o_values[25]), "=f"(o_values[26]), "=f"(o_values[27]), "=f"(o_values[28]), "=f"(o_values[29]), "=f"(o_values[30]), "=f"(o_values[31])
                        : "r"(taddr + 128 + 64 + (unsigned int)(tmem_row_origin << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values[32]), "=f"(o_values[33]), "=f"(o_values[34]), "=f"(o_values[35]), "=f"(o_values[36]), "=f"(o_values[37]), "=f"(o_values[38]), "=f"(o_values[39]), "=f"(o_values[40]), "=f"(o_values[41]), "=f"(o_values[42]), "=f"(o_values[43]), "=f"(o_values[44]), "=f"(o_values[45]), "=f"(o_values[46]), "=f"(o_values[47]), "=f"(o_values[48]), "=f"(o_values[49]), "=f"(o_values[50]), "=f"(o_values[51]), "=f"(o_values[52]), "=f"(o_values[53]), "=f"(o_values[54]), "=f"(o_values[55]), "=f"(o_values[56]), "=f"(o_values[57]), "=f"(o_values[58]), "=f"(o_values[59]), "=f"(o_values[60]), "=f"(o_values[61]), "=f"(o_values[62]), "=f"(o_values[63])
                        : "r"(taddr + 128 + 64 + (unsigned int)(tmem_row_origin << 16) + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_79 = {norm, norm};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values)[_ls], _scale2_79);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++) {
                        o_values[_ls] = o_values[_ls] * norm;
                    }
                    #endif
                    if (row_valid != 0) {
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values[0 + 0], o_values[0 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values[0 + 2], o_values[0 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values[0 + 4], o_values[0 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values[0 + 6], o_values[0 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values[0 + 8], o_values[0 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values[0 + 10], o_values[0 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values[0 + 12], o_values[0 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values[0 + 14], o_values[0 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base + (long long)(o_chunk * 2 * 128) + 64)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values[16 + 0], o_values[16 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values[16 + 2], o_values[16 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values[16 + 4], o_values[16 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values[16 + 6], o_values[16 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values[16 + 8], o_values[16 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values[16 + 10], o_values[16 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values[16 + 12], o_values[16 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values[16 + 14], o_values[16 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base + (long long)(o_chunk * 2 * 128) + 64 + 16)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values[32 + 0], o_values[32 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values[32 + 2], o_values[32 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values[32 + 4], o_values[32 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values[32 + 6], o_values[32 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values[32 + 8], o_values[32 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values[32 + 10], o_values[32 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values[32 + 12], o_values[32 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values[32 + 14], o_values[32 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base + (long long)(o_chunk * 2 * 128) + 64 + 32)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values[48 + 0], o_values[48 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values[48 + 2], o_values[48 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values[48 + 4], o_values[48 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values[48 + 6], o_values[48 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values[48 + 8], o_values[48 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values[48 + 10], o_values[48 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values[48 + 12], o_values[48 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values[48 + 14], o_values[48 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base + (long long)(o_chunk * 2 * 128) + 64 + 48)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                    }
                }
            } else {
                mbarrier_wait_hint(o_full_addr, _phase_o_full_0, 10000000);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp_rows_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values[0]), "=f"(o_values[1]), "=f"(o_values[2]), "=f"(o_values[3]), "=f"(o_values[4]), "=f"(o_values[5]), "=f"(o_values[6]), "=f"(o_values[7]), "=f"(o_values[8]), "=f"(o_values[9]), "=f"(o_values[10]), "=f"(o_values[11]), "=f"(o_values[12]), "=f"(o_values[13]), "=f"(o_values[14]), "=f"(o_values[15]), "=f"(o_values[16]), "=f"(o_values[17]), "=f"(o_values[18]), "=f"(o_values[19]), "=f"(o_values[20]), "=f"(o_values[21]), "=f"(o_values[22]), "=f"(o_values[23]), "=f"(o_values[24]), "=f"(o_values[25]), "=f"(o_values[26]), "=f"(o_values[27]), "=f"(o_values[28]), "=f"(o_values[29]), "=f"(o_values[30]), "=f"(o_values[31])
                        : "r"(taddr + 128 + (unsigned int)(tmem_row_origin << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values[32]), "=f"(o_values[33]), "=f"(o_values[34]), "=f"(o_values[35]), "=f"(o_values[36]), "=f"(o_values[37]), "=f"(o_values[38]), "=f"(o_values[39]), "=f"(o_values[40]), "=f"(o_values[41]), "=f"(o_values[42]), "=f"(o_values[43]), "=f"(o_values[44]), "=f"(o_values[45]), "=f"(o_values[46]), "=f"(o_values[47]), "=f"(o_values[48]), "=f"(o_values[49]), "=f"(o_values[50]), "=f"(o_values[51]), "=f"(o_values[52]), "=f"(o_values[53]), "=f"(o_values[54]), "=f"(o_values[55]), "=f"(o_values[56]), "=f"(o_values[57]), "=f"(o_values[58]), "=f"(o_values[59]), "=f"(o_values[60]), "=f"(o_values[61]), "=f"(o_values[62]), "=f"(o_values[63])
                        : "r"(taddr + 128 + (unsigned int)(tmem_row_origin << 16) + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_80 = {norm, norm};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values)[_ls], _scale2_80);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++) {
                        o_values[_ls] = o_values[_ls] * norm;
                    }
                    #endif
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values[_lp*2 + 0], o_values[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    int o_row_addr = smem_ostage_addr + (unsigned int)(row * 128);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr + (0 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[0])), "r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr + (1 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[4])), "r"(*reinterpret_cast<uint32_t*>(&packed[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(4) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr + (2 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[8])), "r"(*reinterpret_cast<uint32_t*>(&packed[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(8) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr + (3 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[12])), "r"(*reinterpret_cast<uint32_t*>(&packed[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(12) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr + (4 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[16])), "r"(*reinterpret_cast<uint32_t*>(&packed[(16) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(16) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(16) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr + (5 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[20])), "r"(*reinterpret_cast<uint32_t*>(&packed[(20) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(20) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(20) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr + (6 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[24])), "r"(*reinterpret_cast<uint32_t*>(&packed[(24) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(24) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(24) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr + (7 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[28])), "r"(*reinterpret_cast<uint32_t*>(&packed[(28) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(28) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(28) + 3])));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values[0]), "=f"(o_values[1]), "=f"(o_values[2]), "=f"(o_values[3]), "=f"(o_values[4]), "=f"(o_values[5]), "=f"(o_values[6]), "=f"(o_values[7]), "=f"(o_values[8]), "=f"(o_values[9]), "=f"(o_values[10]), "=f"(o_values[11]), "=f"(o_values[12]), "=f"(o_values[13]), "=f"(o_values[14]), "=f"(o_values[15]), "=f"(o_values[16]), "=f"(o_values[17]), "=f"(o_values[18]), "=f"(o_values[19]), "=f"(o_values[20]), "=f"(o_values[21]), "=f"(o_values[22]), "=f"(o_values[23]), "=f"(o_values[24]), "=f"(o_values[25]), "=f"(o_values[26]), "=f"(o_values[27]), "=f"(o_values[28]), "=f"(o_values[29]), "=f"(o_values[30]), "=f"(o_values[31])
                        : "r"(taddr + 128 + 64 + (unsigned int)(tmem_row_origin << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values[32]), "=f"(o_values[33]), "=f"(o_values[34]), "=f"(o_values[35]), "=f"(o_values[36]), "=f"(o_values[37]), "=f"(o_values[38]), "=f"(o_values[39]), "=f"(o_values[40]), "=f"(o_values[41]), "=f"(o_values[42]), "=f"(o_values[43]), "=f"(o_values[44]), "=f"(o_values[45]), "=f"(o_values[46]), "=f"(o_values[47]), "=f"(o_values[48]), "=f"(o_values[49]), "=f"(o_values[50]), "=f"(o_values[51]), "=f"(o_values[52]), "=f"(o_values[53]), "=f"(o_values[54]), "=f"(o_values[55]), "=f"(o_values[56]), "=f"(o_values[57]), "=f"(o_values[58]), "=f"(o_values[59]), "=f"(o_values[60]), "=f"(o_values[61]), "=f"(o_values[62]), "=f"(o_values[63])
                        : "r"(taddr + 128 + 64 + (unsigned int)(tmem_row_origin << 16) + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_81 = {norm, norm};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values)[_ls], _scale2_81);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++) {
                        o_values[_ls] = o_values[_ls] * norm;
                    }
                    #endif
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values[_lp*2 + 0], o_values[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    int o_row_addr_0 = smem_ostage_addr + 16384 + (unsigned int)(row * 128);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0 + (0 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[0])), "r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0 + (1 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[4])), "r"(*reinterpret_cast<uint32_t*>(&packed[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(4) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0 + (2 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[8])), "r"(*reinterpret_cast<uint32_t*>(&packed[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(8) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0 + (3 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[12])), "r"(*reinterpret_cast<uint32_t*>(&packed[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(12) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0 + (4 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[16])), "r"(*reinterpret_cast<uint32_t*>(&packed[(16) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(16) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(16) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0 + (5 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[20])), "r"(*reinterpret_cast<uint32_t*>(&packed[(20) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(20) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(20) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0 + (6 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[24])), "r"(*reinterpret_cast<uint32_t*>(&packed[(24) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(24) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(24) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0 + (7 ^ row % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed[28])), "r"(*reinterpret_cast<uint32_t*>(&packed[(28) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed[(28) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed[(28) + 3])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        tma_store_4d((&tmap_out), o_chunk * 2 * 128, split_idx, head_base, query_idx, smem_ostage_addr);
                        tma_store_4d((&tmap_out), o_chunk * 2 * 128 + 64, split_idx, head_base, query_idx, smem_ostage_addr + 16384);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: compute1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // compute1_main
            const int local_warp_1 = warp - 4;
            int o_chunk_1 = blockIdx.x % 2;
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
            int is_main_1 = 1;
            if (split_idx_1 >= num_main_tiles) {
                is_main_1 = 0;
            }
            int tile_in_table_1 = ((is_main_1 != 0) ? split_idx_1 : split_idx_1 - num_main_tiles);
            int table_width_1 = ((is_main_1 != 0) ? main_width : extra_width);
            int* row_ptr_1 = ((is_main_1 != 0) ? (main_indices + (query_idx_1 * main_index_stride)) : (extra_indices + (query_idx_1 * extra_index_stride)));
            int col_1 = tile_in_table_1 * 128 + row_1;
            int raw_index_1 = -1;
            if (col_1 < table_width_1) {
                raw_index_1 = row_ptr_1[col_1];
            }
            int active_len_1 = table_width_1;
            if (is_main_1 != 0) {
                if (has_main_lengths != 0) {
                    active_len_1 = main_lengths[query_idx_1];
                }
            } else if (has_extra_lengths != 0) {
                active_len_1 = extra_lengths[query_idx_1];
            }
            if (active_len_1 < 0) {
                active_len_1 = 0;
            }
            if (active_len_1 > table_width_1) {
                active_len_1 = table_width_1;
            }
            int valid_1 = 1;
            if (raw_index_1 < 0) {
                valid_1 = 0;
            }
            if (col_1 >= active_len_1) {
                valid_1 = 0;
            }
            uint8_t* cache_1 = ((is_main_1 != 0) ? (main_cache) : (extra_cache));
            int page_shift_1 = ((is_main_1 != 0) ? main_page_shift : extra_page_shift);
            long long page_stride_1 = ((is_main_1 != 0) ? main_page_stride : extra_page_stride);
            int safe_index_1 = ((raw_index_1 >= 0) ? raw_index_1 : 0);
            int page_1 = safe_index_1 >> page_shift_1;
            int slot_in_page_1 = safe_index_1 - (page_1 << page_shift_1);
            int page_size_1 = 1 << page_shift_1;
            long long page_base_1 = (long long)page_1 * page_stride_1;
            long long data_off_1 = page_base_1 + (long long)(slot_in_page_1 * 352);
            long long sf_off_1 = page_base_1 + (long long)(page_size_1 * 352 + slot_in_page_1 * 32);
            int slot_1 = smem_raw_addr + (unsigned int)(row_1 * 400);
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            const int gather_tid_1 = warp * 32 + lane;
            const int g_chunk_1 = gather_tid_1 % 24;
            const int g_row0_1 = gather_tid_1 / 24;
            const int g_is_sf_1 = ((g_chunk_1 >= 22) ? 1 : 0);
            long long g_src_off_1 = (long long)(((g_is_sf_1 != 0) ? 16 * (g_chunk_1 - 14 - 8) : 16 * g_chunk_1));
            int g_dst0_1 = smem_raw_addr + (unsigned int)(g_row0_1 * 400) + (unsigned int)(16 * g_chunk_1);
            unsigned int w4_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)(g_row0_1 * 16)));
            long long od_2 = (long long)w4_1[1] << 32 | (long long)w4_1[0];
            long long osf_1 = (long long)w4_1[3] << 32 | (long long)w4_1[2];
            long long goff_1 = ((g_is_sf_1 != 0) ? osf_1 : od_2);
            if (goff_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1), "l"(cache_1 + (goff_1 + g_src_off_1)));
            }
            unsigned int w4_0_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 16) * 16)));
            long long od_1_1 = (long long)w4_0_1[1] << 32 | (long long)w4_0_1[0];
            long long osf_2_1 = (long long)w4_0_1[3] << 32 | (long long)w4_0_1[2];
            long long goff_3_1 = ((g_is_sf_1 != 0) ? osf_2_1 : od_1_1);
            if (goff_3_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 6400), "l"(cache_1 + (goff_3_1 + g_src_off_1)));
            }
            unsigned int w4_4_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 32) * 16)));
            long long od_5_1 = (long long)w4_4_1[1] << 32 | (long long)w4_4_1[0];
            long long osf_6_1 = (long long)w4_4_1[3] << 32 | (long long)w4_4_1[2];
            long long goff_7_1 = ((g_is_sf_1 != 0) ? osf_6_1 : od_5_1);
            if (goff_7_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 12800), "l"(cache_1 + (goff_7_1 + g_src_off_1)));
            }
            unsigned int w4_8_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_8_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 48) * 16)));
            long long od_9_1 = (long long)w4_8_1[1] << 32 | (long long)w4_8_1[0];
            long long osf_10_1 = (long long)w4_8_1[3] << 32 | (long long)w4_8_1[2];
            long long goff_11_1 = ((g_is_sf_1 != 0) ? osf_10_1 : od_9_1);
            if (goff_11_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 19200), "l"(cache_1 + (goff_11_1 + g_src_off_1)));
            }
            unsigned int w4_12_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 64) * 16)));
            long long od_13_1 = (long long)w4_12_1[1] << 32 | (long long)w4_12_1[0];
            long long osf_14_1 = (long long)w4_12_1[3] << 32 | (long long)w4_12_1[2];
            long long goff_15_1 = ((g_is_sf_1 != 0) ? osf_14_1 : od_13_1);
            if (goff_15_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 25600), "l"(cache_1 + (goff_15_1 + g_src_off_1)));
            }
            unsigned int w4_16_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_16_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 80) * 16)));
            long long od_17_1 = (long long)w4_16_1[1] << 32 | (long long)w4_16_1[0];
            long long osf_18_1 = (long long)w4_16_1[3] << 32 | (long long)w4_16_1[2];
            long long goff_19_1 = ((g_is_sf_1 != 0) ? osf_18_1 : od_17_1);
            if (goff_19_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 32000), "l"(cache_1 + (goff_19_1 + g_src_off_1)));
            }
            unsigned int w4_20_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_20_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 96) * 16)));
            long long od_21_1 = (long long)w4_20_1[1] << 32 | (long long)w4_20_1[0];
            long long osf_22_1 = (long long)w4_20_1[3] << 32 | (long long)w4_20_1[2];
            long long goff_23_1 = ((g_is_sf_1 != 0) ? osf_22_1 : od_21_1);
            if (goff_23_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 38400), "l"(cache_1 + (goff_23_1 + g_src_off_1)));
            }
            unsigned int w4_24_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_24_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 112) * 16)));
            long long od_25_1 = (long long)w4_24_1[1] << 32 | (long long)w4_24_1[0];
            long long osf_26_1 = (long long)w4_24_1[3] << 32 | (long long)w4_24_1[2];
            long long goff_27_1 = ((g_is_sf_1 != 0) ? osf_26_1 : od_25_1);
            if (goff_27_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 44800), "l"(cache_1 + (goff_27_1 + g_src_off_1)));
            }
            asm volatile("cp.async.commit_group;");
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
            for (int i_1 = 0; i_1 < 3; i_1++) {
                int unit_1 = q_warp_1 + 12 * i_1;
                if (unit_1 < 28) {
                    int q_block_1 = unit_1 / 7;
                    int kset_1 = unit_1 - q_block_1 * 7;
                    int q_row_1 = q_block_1 * 32 + lane;
                    if (head_base_1 + q_block_1 * 32 < num_heads) {
                        int q_row_addr_1 = smem_qstage_addr + (unsigned int)(kset_1 * 16384) + (unsigned int)(q_row_1 * 128);
                        unsigned int sf_word_2 = 0;
                        for (int bp_1 = 0; bp_1 < 2; bp_1++) {
                            unsigned int words_1[4];
                            unsigned int qa_1[4];
                            unsigned int qb_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_1[(0) + 3]))
                                : "r"(q_row_addr_1 + (4 * bp_1 ^ q_row_1 % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_2[(0) + 3]))
                                : "r"(q_row_addr_1 + (4 * bp_1 + 1 ^ q_row_1 % 8) * 16));
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
                            float _fabs_32 = fabsf(qv_1[0]);
                            float _fabs_33 = fabsf(qv_1[1]);
                            float _max_33 = max_noftz(_fabs_32, _fabs_33);
                            m8_1[0] = _max_33;
                            float _fabs_34 = fabsf(qv_1[2]);
                            float _fabs_35 = fabsf(qv_1[3]);
                            float _max_34 = max_noftz(_fabs_34, _fabs_35);
                            m8_1[1] = _max_34;
                            float _fabs_36 = fabsf(qv_1[4]);
                            float _fabs_37 = fabsf(qv_1[5]);
                            float _max_35 = max_noftz(_fabs_36, _fabs_37);
                            m8_1[2] = _max_35;
                            float _fabs_38 = fabsf(qv_1[6]);
                            float _fabs_39 = fabsf(qv_1[7]);
                            float _max_36 = max_noftz(_fabs_38, _fabs_39);
                            m8_1[3] = _max_36;
                            float _fabs_40 = fabsf(qv_1[8]);
                            float _fabs_41 = fabsf(qv_1[9]);
                            float _max_37 = max_noftz(_fabs_40, _fabs_41);
                            m8_1[4] = _max_37;
                            float _fabs_42 = fabsf(qv_1[10]);
                            float _fabs_43 = fabsf(qv_1[11]);
                            float _max_38 = max_noftz(_fabs_42, _fabs_43);
                            m8_1[5] = _max_38;
                            float _fabs_44 = fabsf(qv_1[12]);
                            float _fabs_45 = fabsf(qv_1[13]);
                            float _max_39 = max_noftz(_fabs_44, _fabs_45);
                            m8_1[6] = _max_39;
                            float _fabs_46 = fabsf(qv_1[14]);
                            float _fabs_47 = fabsf(qv_1[15]);
                            float _max_40 = max_noftz(_fabs_46, _fabs_47);
                            m8_1[7] = _max_40;
                            float m4_1[4];
                            float _max_41 = max_noftz(m8_1[0], m8_1[1]);
                            m4_1[0] = _max_41;
                            float _max_42 = max_noftz(m8_1[2], m8_1[3]);
                            m4_1[1] = _max_42;
                            float _max_43 = max_noftz(m8_1[4], m8_1[5]);
                            m4_1[2] = _max_43;
                            float _max_44 = max_noftz(m8_1[6], m8_1[7]);
                            m4_1[3] = _max_44;
                            float _max_45 = max_noftz(m4_1[0], m4_1[1]);
                            float _max_46 = max_noftz(m4_1[2], m4_1[3]);
                            float _max_47 = max_noftz(_max_45, _max_46);
                            float amax_1 = _max_47;
                            float sc_1 = amax_1 * inv_six_1;
                            uint16_t _e4m3x2_f32_10;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_10) : "f"(0.0f), "f"(sc_1));
                            uint16_t sc_pair_1 = _e4m3x2_f32_10;
                            unsigned int sc_byte_1 = (unsigned int)sc_pair_1 & 255;
                            unsigned int sc_exp_1 = sc_byte_1 >> 3 & 15;
                            unsigned int sc_man_1 = sc_byte_1 & 7;
                            float inv_1 = 0.0f;
                            if (sc_exp_1 == 0) {
                                inv_1 = __uint_as_float(smem_rcptab[8 + sc_man_1]) * 512.0f;
                            } else {
                                inv_1 = __uint_as_float(smem_rcptab[sc_man_1]) * __uint_as_float(134 - sc_exp_1 << 23);
                            }
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
                            uint32_t _fp4_pair_16;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_16) : "f"(qv_1[0]), "f"(qv_1[1]));
                            uint32_t _fp4_pair_17;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_17) : "f"(qv_1[2]), "f"(qv_1[3]));
                            uint32_t _fp4_pair_18;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_18) : "f"(qv_1[4]), "f"(qv_1[5]));
                            uint32_t _fp4_pair_19;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_19) : "f"(qv_1[6]), "f"(qv_1[7]));
                            uint32_t _fp4_pair_20;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_20) : "f"(qv_1[8]), "f"(qv_1[9]));
                            uint32_t _fp4_pair_21;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_21) : "f"(qv_1[10]), "f"(qv_1[11]));
                            uint32_t _fp4_pair_22;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_22) : "f"(qv_1[12]), "f"(qv_1[13]));
                            uint32_t _fp4_pair_23;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_23) : "f"(qv_1[14]), "f"(qv_1[15]));
                            words_1[0] = _fp4_pair_16 | _fp4_pair_17 << 8 | _fp4_pair_18 << 16 | _fp4_pair_19 << 24;
                            words_1[1] = _fp4_pair_20 | _fp4_pair_21 << 8 | _fp4_pair_22 << 16 | _fp4_pair_23 << 24;
                            sf_word_2 = sf_word_2 | sc_byte_1 << (unsigned int)(8 * (2 * bp_1));
                            unsigned int qa_0_1[4];
                            unsigned int qb_1_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_1[(0) + 3]))
                                : "r"(q_row_addr_1 + (4 * bp_1 + 2 ^ q_row_1 % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_1[(0) + 3]))
                                : "r"(q_row_addr_1 + (4 * bp_1 + 2 + 1 ^ q_row_1 % 8) * 16));
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
                            float _fabs_48 = fabsf(qv_2_1[0]);
                            float _fabs_49 = fabsf(qv_2_1[1]);
                            float _max_48 = max_noftz(_fabs_48, _fabs_49);
                            m8_3_1[0] = _max_48;
                            float _fabs_50 = fabsf(qv_2_1[2]);
                            float _fabs_51 = fabsf(qv_2_1[3]);
                            float _max_49 = max_noftz(_fabs_50, _fabs_51);
                            m8_3_1[1] = _max_49;
                            float _fabs_52 = fabsf(qv_2_1[4]);
                            float _fabs_53 = fabsf(qv_2_1[5]);
                            float _max_50 = max_noftz(_fabs_52, _fabs_53);
                            m8_3_1[2] = _max_50;
                            float _fabs_54 = fabsf(qv_2_1[6]);
                            float _fabs_55 = fabsf(qv_2_1[7]);
                            float _max_51 = max_noftz(_fabs_54, _fabs_55);
                            m8_3_1[3] = _max_51;
                            float _fabs_56 = fabsf(qv_2_1[8]);
                            float _fabs_57 = fabsf(qv_2_1[9]);
                            float _max_52 = max_noftz(_fabs_56, _fabs_57);
                            m8_3_1[4] = _max_52;
                            float _fabs_58 = fabsf(qv_2_1[10]);
                            float _fabs_59 = fabsf(qv_2_1[11]);
                            float _max_53 = max_noftz(_fabs_58, _fabs_59);
                            m8_3_1[5] = _max_53;
                            float _fabs_60 = fabsf(qv_2_1[12]);
                            float _fabs_61 = fabsf(qv_2_1[13]);
                            float _max_54 = max_noftz(_fabs_60, _fabs_61);
                            m8_3_1[6] = _max_54;
                            float _fabs_62 = fabsf(qv_2_1[14]);
                            float _fabs_63 = fabsf(qv_2_1[15]);
                            float _max_55 = max_noftz(_fabs_62, _fabs_63);
                            m8_3_1[7] = _max_55;
                            float m4_4_1[4];
                            float _max_56 = max_noftz(m8_3_1[0], m8_3_1[1]);
                            m4_4_1[0] = _max_56;
                            float _max_57 = max_noftz(m8_3_1[2], m8_3_1[3]);
                            m4_4_1[1] = _max_57;
                            float _max_58 = max_noftz(m8_3_1[4], m8_3_1[5]);
                            m4_4_1[2] = _max_58;
                            float _max_59 = max_noftz(m8_3_1[6], m8_3_1[7]);
                            m4_4_1[3] = _max_59;
                            float _max_60 = max_noftz(m4_4_1[0], m4_4_1[1]);
                            float _max_61 = max_noftz(m4_4_1[2], m4_4_1[3]);
                            float _max_62 = max_noftz(_max_60, _max_61);
                            float amax_5_1 = _max_62;
                            float sc_6_1 = amax_5_1 * inv_six_1;
                            uint16_t _e4m3x2_f32_11;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_11) : "f"(0.0f), "f"(sc_6_1));
                            uint16_t sc_pair_7_1 = _e4m3x2_f32_11;
                            unsigned int sc_byte_8_1 = (unsigned int)sc_pair_7_1 & 255;
                            unsigned int sc_exp_9_1 = sc_byte_8_1 >> 3 & 15;
                            unsigned int sc_man_10_1 = sc_byte_8_1 & 7;
                            float inv_11_1 = 0.0f;
                            if (sc_exp_9_1 == 0) {
                                inv_11_1 = __uint_as_float(smem_rcptab[8 + sc_man_10_1]) * 512.0f;
                            } else {
                                inv_11_1 = __uint_as_float(smem_rcptab[sc_man_10_1]) * __uint_as_float(134 - sc_exp_9_1 << 23);
                            }
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_1 = {inv_11_1, inv_11_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2_1)[_ls], _scale2_1);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                qv_2_1[_ls] = qv_2_1[_ls] * inv_11_1;
                            }
                            #endif
                            uint32_t _fp4_pair_24;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_24) : "f"(qv_2_1[0]), "f"(qv_2_1[1]));
                            uint32_t _fp4_pair_25;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_25) : "f"(qv_2_1[2]), "f"(qv_2_1[3]));
                            uint32_t _fp4_pair_26;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_26) : "f"(qv_2_1[4]), "f"(qv_2_1[5]));
                            uint32_t _fp4_pair_27;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_27) : "f"(qv_2_1[6]), "f"(qv_2_1[7]));
                            uint32_t _fp4_pair_28;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_28) : "f"(qv_2_1[8]), "f"(qv_2_1[9]));
                            uint32_t _fp4_pair_29;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_29) : "f"(qv_2_1[10]), "f"(qv_2_1[11]));
                            uint32_t _fp4_pair_30;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_30) : "f"(qv_2_1[12]), "f"(qv_2_1[13]));
                            uint32_t _fp4_pair_31;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_31) : "f"(qv_2_1[14]), "f"(qv_2_1[15]));
                            words_1[2] = _fp4_pair_24 | _fp4_pair_25 << 8 | _fp4_pair_26 << 16 | _fp4_pair_27 << 24;
                            words_1[3] = _fp4_pair_28 | _fp4_pair_29 << 8 | _fp4_pair_30 << 16 | _fp4_pair_31 << 24;
                            sf_word_2 = sf_word_2 | sc_byte_8_1 << (unsigned int)(8 * (2 * bp_1 + 1));
                            int chunk_1 = 2 * kset_1 + bp_1;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_qf4_addr + (unsigned int)(chunk_1 / 8 * 16384 + (q_row_1 * 128 + (chunk_1 % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3])));
                        }
                        smem_qsf32[kset_1 / 4 * 2048 + q_row_1 % 32 / 8 * 512 + kset_1 % 4 * 128 + q_row_1 % 8 * 16 + q_row_1 / 32 % 4 * 4 >> 2] = sf_word_2;
                    } else {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(2 * kset_1 / 8 * 16384 + (q_row_1 * 128 + (2 * kset_1 % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)((2 * kset_1 + 1) / 8 * 16384 + (q_row_1 * 128 + ((2 * kset_1 + 1) % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        smem_qsf32[kset_1 / 4 * 2048 + q_row_1 % 32 / 8 * 512 + kset_1 % 4 * 128 + q_row_1 % 8 * 16 + q_row_1 / 32 % 4 * 4 >> 2] = 0;
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            unsigned int _phase_q_ready_0_1 = 0;
            mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0_1, 10000000);
            _phase_q_ready_0_1 ^= 1;
            asm volatile("cp.async.wait_group 0;");
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            int nope_hi_v_1 = 14;
            int rope_pairs_hi_v_1 = 1;
            if (valid_1 != 0) {
                {
                    unsigned int sfw_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 3]))
                        : "r"(slot_1 + 352 + 16));
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_1[0];
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_1[1];
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_1[2];
                    {
                        smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    }
                    for (int c_2 = 8; c_2 < nope_hi_v_1; c_2++) {
                        unsigned int raw_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 3]))
                            : "r"(slot_1 + 16 * c_2));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_kf4_addr + (unsigned int)(c_2 / 8 * 16384 + (row_1 * 128 + (c_2 % 8 * 16 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&raw_1[0])), "r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 3])));
                        unsigned int sf_word_3 = smem_raw32[row_1 * 400 + 352 + 2 * c_2 >> 2];
                        int block_1 = 2 * c_2 - 16 * o_chunk_1;
                        unsigned int scale_2 = sf_word_3 >> (unsigned int)(8 * ((c_2 & 1) * 2)) & 255;
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
                        if (c_2 * 2 >> 4 == o_chunk_1) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_v_addr + (unsigned int)(block_1 / 8 * 16384 + (row_1 * 128 + (block_1 % 8 * 16 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 3])));
                        }
                        int block_0_1 = 2 * c_2 + 1 - 16 * o_chunk_1;
                        unsigned int scale_1_1 = sf_word_3 >> (unsigned int)(8 * ((c_2 & 1) * 2 + 1)) & 255;
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
                        if (c_2 * 2 >> 4 == o_chunk_1) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_v_addr + (unsigned int)(block_0_1 / 8 * 16384 + (row_1 * 128 + (block_0_1 % 8 * 16 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_2_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2_1[(0) + 3])));
                        }
                    }
                }
                {
                    for (int jp = 0; jp < rope_pairs_hi_v_1; jp++) {
                        unsigned int vrope[4];
                        int j = 2 * jp;
                        unsigned int rope[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 3]))
                            : "r"(slot_1 + 224 + 16 * j));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_krope_addr + (unsigned int)(row_1 * 128 + (j * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&rope[0])), "r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 3])));
                        float lo = __uint_as_float(rope[0] << 16);
                        float hi = __uint_as_float(rope[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_12;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_12) : "f"(hi), "f"(lo));
                        uint16_t pair = _e4m3x2_f32_12;
                        {
                            vrope[0] = (unsigned int)pair;
                        }
                        float lo_0 = __uint_as_float(rope[1] << 16);
                        float hi_1 = __uint_as_float(rope[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_13;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_13) : "f"(hi_1), "f"(lo_0));
                        uint16_t pair_2 = _e4m3x2_f32_13;
                        {
                            vrope[0] = vrope[0] | (unsigned int)pair_2 << 16;
                        }
                        float lo_3 = __uint_as_float(rope[2] << 16);
                        float hi_4 = __uint_as_float(rope[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_14;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_14) : "f"(hi_4), "f"(lo_3));
                        uint16_t pair_5 = _e4m3x2_f32_14;
                        {
                            vrope[1] = (unsigned int)pair_5;
                        }
                        float lo_6 = __uint_as_float(rope[3] << 16);
                        float hi_7 = __uint_as_float(rope[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_15;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_15) : "f"(hi_7), "f"(lo_6));
                        uint16_t pair_8 = _e4m3x2_f32_15;
                        {
                            vrope[1] = vrope[1] | (unsigned int)pair_8 << 16;
                        }
                        int j_9 = 2 * jp + 1;
                        unsigned int rope_10[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10[(0) + 3]))
                            : "r"(slot_1 + 224 + 16 * j_9));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_krope_addr + (unsigned int)(row_1 * 128 + (j_9 * 16 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&rope_10[0])), "r"(*reinterpret_cast<uint32_t*>(&rope_10[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&rope_10[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&rope_10[(0) + 3])));
                        float lo_11 = __uint_as_float(rope_10[0] << 16);
                        float hi_12 = __uint_as_float(rope_10[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_16;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_16) : "f"(hi_12), "f"(lo_11));
                        uint16_t pair_13 = _e4m3x2_f32_16;
                        {
                            vrope[2] = (unsigned int)pair_13;
                        }
                        float lo_14 = __uint_as_float(rope_10[1] << 16);
                        float hi_15 = __uint_as_float(rope_10[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_17;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_17) : "f"(hi_15), "f"(lo_14));
                        uint16_t pair_16 = _e4m3x2_f32_17;
                        {
                            vrope[2] = vrope[2] | (unsigned int)pair_16 << 16;
                        }
                        float lo_17 = __uint_as_float(rope_10[2] << 16);
                        float hi_18 = __uint_as_float(rope_10[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_18) : "f"(hi_18), "f"(lo_17));
                        uint16_t pair_19 = _e4m3x2_f32_18;
                        {
                            vrope[3] = (unsigned int)pair_19;
                        }
                        float lo_20 = __uint_as_float(rope_10[3] << 16);
                        float hi_21 = __uint_as_float(rope_10[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_19;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_19) : "f"(hi_21), "f"(lo_20));
                        uint16_t pair_22 = _e4m3x2_f32_19;
                        {
                            vrope[3] = vrope[3] | (unsigned int)pair_22 << 16;
                        }
                        if (o_chunk_1 == 1) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_v_addr + (unsigned int)((jp - 4 + 16) / 8 * 16384 + (row_1 * 128 + ((jp - 4 + 16) % 8 * 16 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&vrope[0])), "r"(*reinterpret_cast<uint32_t*>(&vrope[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vrope[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vrope[(0) + 3])));
                        }
                    }
                }
            } else {
                {
                    for (int c_3 = 8; c_3 < nope_hi_v_1; c_3++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kf4_addr + (unsigned int)(c_3 / 8 * 16384 + (row_1 * 128 + (c_3 % 8 * 16 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        if (c_3 * 2 >> 4 == o_chunk_1) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)((2 * c_3 - 16 * o_chunk_1) / 8 * 16384 + (row_1 * 128 + ((2 * c_3 - 16 * o_chunk_1) % 8 * 16 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)((2 * c_3 + 1 - 16 * o_chunk_1) / 8 * 16384 + (row_1 * 128 + ((2 * c_3 + 1 - 16 * o_chunk_1) % 8 * 16 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        }
                    }
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                }
                {
                    for (int jp_1 = 0; jp_1 < rope_pairs_hi_v_1; jp_1++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_krope_addr + (unsigned int)(row_1 * 128 + (2 * jp_1 * 16 ^ row_1 % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_krope_addr + (unsigned int)(row_1 * 128 + ((2 * jp_1 + 1) * 16 ^ row_1 % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        if (o_chunk_1 == 1) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)((jp_1 - 4 + 16) / 8 * 16384 + (row_1 * 128 + ((jp_1 - 4 + 16) % 8 * 16 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        }
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_1 = bmm2_scale[0];
            float sink_log2_1 = 0.0f;
            int has_sink_row_1 = 0;
            if (has_sinks != 0 && split_idx_1 == 0 && row_valid_1 != 0) {
                has_sink_row_1 = 1;
                sink_log2_1 = sinks[head_row_1] * 1.4426950408889634f;
            }
            unsigned int _phase_s_full_0_1 = 0;
            mbarrier_wait_hint(s_full_addr, _phase_s_full_0_1, 10000000);
            _phase_s_full_0_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float slice_max_1 = -CAKE_INF;
            float score_values_1[32];
            unsigned int mask_words_1[4];
            mask_words_1[0] = smem_mask[0];
            mask_words_1[1] = smem_mask[1];
            mask_words_1[2] = smem_mask[2];
            mask_words_1[3] = smem_mask[3];
            if (warp_rows_valid_1 != 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(score_values_1[0]), "=f"(score_values_1[1]), "=f"(score_values_1[2]), "=f"(score_values_1[3]), "=f"(score_values_1[4]), "=f"(score_values_1[5]), "=f"(score_values_1[6]), "=f"(score_values_1[7]), "=f"(score_values_1[8]), "=f"(score_values_1[9]), "=f"(score_values_1[10]), "=f"(score_values_1[11]), "=f"(score_values_1[12]), "=f"(score_values_1[13]), "=f"(score_values_1[14]), "=f"(score_values_1[15]), "=f"(score_values_1[16]), "=f"(score_values_1[17]), "=f"(score_values_1[18]), "=f"(score_values_1[19]), "=f"(score_values_1[20]), "=f"(score_values_1[21]), "=f"(score_values_1[22]), "=f"(score_values_1[23]), "=f"(score_values_1[24]), "=f"(score_values_1[25]), "=f"(score_values_1[26]), "=f"(score_values_1[27]), "=f"(score_values_1[28]), "=f"(score_values_1[29]), "=f"(score_values_1[30]), "=f"(score_values_1[31])
                    : "r"(taddr + 64 + (unsigned int)(tmem_row_origin_1 << 16)));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int full_mask_1 = mask_words_1[2];
                if (full_mask_1 != 4294967295u) {
                    if ((mask_words_1[2] & 1) == 0) {
                        score_values_1[0] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 1 & 1) == 0) {
                        score_values_1[1] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 2 & 1) == 0) {
                        score_values_1[2] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 3 & 1) == 0) {
                        score_values_1[3] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 4 & 1) == 0) {
                        score_values_1[4] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 5 & 1) == 0) {
                        score_values_1[5] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 6 & 1) == 0) {
                        score_values_1[6] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 7 & 1) == 0) {
                        score_values_1[7] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 8 & 1) == 0) {
                        score_values_1[8] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 9 & 1) == 0) {
                        score_values_1[9] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 10 & 1) == 0) {
                        score_values_1[10] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 11 & 1) == 0) {
                        score_values_1[11] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 12 & 1) == 0) {
                        score_values_1[12] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 13 & 1) == 0) {
                        score_values_1[13] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 14 & 1) == 0) {
                        score_values_1[14] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 15 & 1) == 0) {
                        score_values_1[15] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 16 & 1) == 0) {
                        score_values_1[16] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 17 & 1) == 0) {
                        score_values_1[17] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 18 & 1) == 0) {
                        score_values_1[18] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 19 & 1) == 0) {
                        score_values_1[19] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 20 & 1) == 0) {
                        score_values_1[20] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 21 & 1) == 0) {
                        score_values_1[21] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 22 & 1) == 0) {
                        score_values_1[22] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 23 & 1) == 0) {
                        score_values_1[23] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 24 & 1) == 0) {
                        score_values_1[24] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 25 & 1) == 0) {
                        score_values_1[25] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 26 & 1) == 0) {
                        score_values_1[26] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 27 & 1) == 0) {
                        score_values_1[27] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 28 & 1) == 0) {
                        score_values_1[28] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 29 & 1) == 0) {
                        score_values_1[29] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 30 & 1) == 0) {
                        score_values_1[30] = -CAKE_INF;
                    }
                    if ((mask_words_1[2] >> 31 & 1) == 0) {
                        score_values_1[31] = -CAKE_INF;
                    }
                }
                float2 _reg_reduce_max2_10 = {-CAKE_INF, -CAKE_INF};
                _reg_reduce_max2_10.x = max_noftz(_reg_reduce_max2_10.x, max_noftz(score_values_1[0], score_values_1[1]));
                _reg_reduce_max2_10.y = max_noftz(_reg_reduce_max2_10.y, max_noftz(score_values_1[2], score_values_1[3]));
                _reg_reduce_max2_10.x = max_noftz(_reg_reduce_max2_10.x, max_noftz(score_values_1[4], score_values_1[5]));
                _reg_reduce_max2_10.y = max_noftz(_reg_reduce_max2_10.y, max_noftz(score_values_1[6], score_values_1[7]));
                _reg_reduce_max2_10.x = max_noftz(_reg_reduce_max2_10.x, max_noftz(score_values_1[8], score_values_1[9]));
                _reg_reduce_max2_10.y = max_noftz(_reg_reduce_max2_10.y, max_noftz(score_values_1[10], score_values_1[11]));
                _reg_reduce_max2_10.x = max_noftz(_reg_reduce_max2_10.x, max_noftz(score_values_1[12], score_values_1[13]));
                _reg_reduce_max2_10.y = max_noftz(_reg_reduce_max2_10.y, max_noftz(score_values_1[14], score_values_1[15]));
                _reg_reduce_max2_10.x = max_noftz(_reg_reduce_max2_10.x, max_noftz(score_values_1[16], score_values_1[17]));
                _reg_reduce_max2_10.y = max_noftz(_reg_reduce_max2_10.y, max_noftz(score_values_1[18], score_values_1[19]));
                _reg_reduce_max2_10.x = max_noftz(_reg_reduce_max2_10.x, max_noftz(score_values_1[20], score_values_1[21]));
                _reg_reduce_max2_10.y = max_noftz(_reg_reduce_max2_10.y, max_noftz(score_values_1[22], score_values_1[23]));
                _reg_reduce_max2_10.x = max_noftz(_reg_reduce_max2_10.x, max_noftz(score_values_1[24], score_values_1[25]));
                _reg_reduce_max2_10.y = max_noftz(_reg_reduce_max2_10.y, max_noftz(score_values_1[26], score_values_1[27]));
                _reg_reduce_max2_10.x = max_noftz(_reg_reduce_max2_10.x, max_noftz(score_values_1[28], score_values_1[29]));
                _reg_reduce_max2_10.y = max_noftz(_reg_reduce_max2_10.y, max_noftz(score_values_1[30], score_values_1[31]));
                float score_values_max_1 = row_max_reduce(_reg_reduce_max2_10);
                slice_max_1 = score_values_max_1;
            }
            smem_pmax[128 + row_1] = slice_max_1;
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            float _max_63 = max_noftz(smem_pmax[row_1], smem_pmax[128 + row_1]);
            float _max_64 = max_noftz(_max_63, smem_pmax[256 + row_1]);
            float tile_max_1 = _max_64;
            float row_max_scaled_1 = tile_max_1 * softmax_scale_log2_1;
            if (has_sink_row_1 != 0) {
                float _max_65 = max_noftz(row_max_scaled_1, sink_log2_1);
                row_max_scaled_1 = _max_65;
            }
            if (row_max_scaled_1 == -CAKE_INF) {
                row_max_scaled_1 = 0.0f;
            }
            float slice_sum_1 = 0.0f;
            float rsum_1 = 0.0f;
            if (warp_rows_valid_1 != 0) {
                float score_bias_1 = -row_max_scaled_1;
                const float2 _fma_b2_11 = {softmax_scale_log2_1, softmax_scale_log2_1};
                const float2 _fma_c2_12 = {score_bias_1, score_bias_1};
                float2 _fma_pair_13 = fma_f32x2(make_float2(score_values_1[0], score_values_1[1]), _fma_b2_11, _fma_c2_12);
                score_values_1[0] = _fma_pair_13.x;
                score_values_1[1] = _fma_pair_13.y;
                float2 _fma_pair_14 = fma_f32x2(make_float2(score_values_1[2], score_values_1[3]), _fma_b2_11, _fma_c2_12);
                score_values_1[2] = _fma_pair_14.x;
                score_values_1[3] = _fma_pair_14.y;
                float2 _fma_pair_15 = fma_f32x2(make_float2(score_values_1[4], score_values_1[5]), _fma_b2_11, _fma_c2_12);
                score_values_1[4] = _fma_pair_15.x;
                score_values_1[5] = _fma_pair_15.y;
                float2 _fma_pair_16 = fma_f32x2(make_float2(score_values_1[6], score_values_1[7]), _fma_b2_11, _fma_c2_12);
                score_values_1[6] = _fma_pair_16.x;
                score_values_1[7] = _fma_pair_16.y;
                float2 _fma_pair_17 = fma_f32x2(make_float2(score_values_1[8], score_values_1[9]), _fma_b2_11, _fma_c2_12);
                score_values_1[8] = _fma_pair_17.x;
                score_values_1[9] = _fma_pair_17.y;
                float2 _fma_pair_18 = fma_f32x2(make_float2(score_values_1[10], score_values_1[11]), _fma_b2_11, _fma_c2_12);
                score_values_1[10] = _fma_pair_18.x;
                score_values_1[11] = _fma_pair_18.y;
                float2 _fma_pair_19 = fma_f32x2(make_float2(score_values_1[12], score_values_1[13]), _fma_b2_11, _fma_c2_12);
                score_values_1[12] = _fma_pair_19.x;
                score_values_1[13] = _fma_pair_19.y;
                float2 _fma_pair_20 = fma_f32x2(make_float2(score_values_1[14], score_values_1[15]), _fma_b2_11, _fma_c2_12);
                score_values_1[14] = _fma_pair_20.x;
                score_values_1[15] = _fma_pair_20.y;
                float2 _fma_pair_21 = fma_f32x2(make_float2(score_values_1[16], score_values_1[17]), _fma_b2_11, _fma_c2_12);
                score_values_1[16] = _fma_pair_21.x;
                score_values_1[17] = _fma_pair_21.y;
                float2 _fma_pair_22 = fma_f32x2(make_float2(score_values_1[18], score_values_1[19]), _fma_b2_11, _fma_c2_12);
                score_values_1[18] = _fma_pair_22.x;
                score_values_1[19] = _fma_pair_22.y;
                float2 _fma_pair_23 = fma_f32x2(make_float2(score_values_1[20], score_values_1[21]), _fma_b2_11, _fma_c2_12);
                score_values_1[20] = _fma_pair_23.x;
                score_values_1[21] = _fma_pair_23.y;
                float2 _fma_pair_24 = fma_f32x2(make_float2(score_values_1[22], score_values_1[23]), _fma_b2_11, _fma_c2_12);
                score_values_1[22] = _fma_pair_24.x;
                score_values_1[23] = _fma_pair_24.y;
                float2 _fma_pair_25 = fma_f32x2(make_float2(score_values_1[24], score_values_1[25]), _fma_b2_11, _fma_c2_12);
                score_values_1[24] = _fma_pair_25.x;
                score_values_1[25] = _fma_pair_25.y;
                float2 _fma_pair_26 = fma_f32x2(make_float2(score_values_1[26], score_values_1[27]), _fma_b2_11, _fma_c2_12);
                score_values_1[26] = _fma_pair_26.x;
                score_values_1[27] = _fma_pair_26.y;
                float2 _fma_pair_27 = fma_f32x2(make_float2(score_values_1[28], score_values_1[29]), _fma_b2_11, _fma_c2_12);
                score_values_1[28] = _fma_pair_27.x;
                score_values_1[29] = _fma_pair_27.y;
                float2 _fma_pair_28 = fma_f32x2(make_float2(score_values_1[30], score_values_1[31]), _fma_b2_11, _fma_c2_12);
                score_values_1[30] = _fma_pair_28.x;
                score_values_1[31] = _fma_pair_28.y;
                #pragma unroll
                for (int _le = 0; _le < 32; _le++) {
                    score_values_1[_le] = approx_exp2(score_values_1[_le]);
                }
                float2 _reg_reduce_sum2_29 = make_float2(0.0f, 0.0f);
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[0], score_values_1[1]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[2], score_values_1[3]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[4], score_values_1[5]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[6], score_values_1[7]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[8], score_values_1[9]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[10], score_values_1[11]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[12], score_values_1[13]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[14], score_values_1[15]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[16], score_values_1[17]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[18], score_values_1[19]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[20], score_values_1[21]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[22], score_values_1[23]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[24], score_values_1[25]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[26], score_values_1[27]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[28], score_values_1[29]));
                _reg_reduce_sum2_29 = add_f32x2(_reg_reduce_sum2_29, make_float2(score_values_1[30], score_values_1[31]));
                float score_values_sum_1 = _reg_reduce_sum2_29.x + _reg_reduce_sum2_29.y;
                slice_sum_1 = score_values_sum_1;
                if (row_valid_1 == 0) {
                    score_values_1[0] = 0.0f;
                    score_values_1[1] = 0.0f;
                    score_values_1[2] = 0.0f;
                    score_values_1[3] = 0.0f;
                    score_values_1[4] = 0.0f;
                    score_values_1[5] = 0.0f;
                    score_values_1[6] = 0.0f;
                    score_values_1[7] = 0.0f;
                    score_values_1[8] = 0.0f;
                    score_values_1[9] = 0.0f;
                    score_values_1[10] = 0.0f;
                    score_values_1[11] = 0.0f;
                    score_values_1[12] = 0.0f;
                    score_values_1[13] = 0.0f;
                    score_values_1[14] = 0.0f;
                    score_values_1[15] = 0.0f;
                    score_values_1[16] = 0.0f;
                    score_values_1[17] = 0.0f;
                    score_values_1[18] = 0.0f;
                    score_values_1[19] = 0.0f;
                    score_values_1[20] = 0.0f;
                    score_values_1[21] = 0.0f;
                    score_values_1[22] = 0.0f;
                    score_values_1[23] = 0.0f;
                    score_values_1[24] = 0.0f;
                    score_values_1[25] = 0.0f;
                    score_values_1[26] = 0.0f;
                    score_values_1[27] = 0.0f;
                    score_values_1[28] = 0.0f;
                    score_values_1[29] = 0.0f;
                    score_values_1[30] = 0.0f;
                    score_values_1[31] = 0.0f;
                }
                unsigned int packed_p_1[8];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(score_values_1[0]), "f"(score_values_1[1]),
                                           "f"(score_values_1[2]), "f"(score_values_1[3]));
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
                        : "=r"(_packed) : "f"(score_values_1[4]), "f"(score_values_1[5]),
                                           "f"(score_values_1[6]), "f"(score_values_1[7]));
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
                        : "=r"(_packed) : "f"(score_values_1[8]), "f"(score_values_1[9]),
                                           "f"(score_values_1[10]), "f"(score_values_1[11]));
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
                        : "=r"(_packed) : "f"(score_values_1[12]), "f"(score_values_1[13]),
                                           "f"(score_values_1[14]), "f"(score_values_1[15]));
                    packed_p_1[3] = _packed;
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
                        : "=r"(_packed) : "f"(score_values_1[16]), "f"(score_values_1[17]),
                                           "f"(score_values_1[18]), "f"(score_values_1[19]));
                    packed_p_1[4] = _packed;
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
                        : "=r"(_packed) : "f"(score_values_1[20]), "f"(score_values_1[21]),
                                           "f"(score_values_1[22]), "f"(score_values_1[23]));
                    packed_p_1[5] = _packed;
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
                        : "=r"(_packed) : "f"(score_values_1[24]), "f"(score_values_1[25]),
                                           "f"(score_values_1[26]), "f"(score_values_1[27]));
                    packed_p_1[6] = _packed;
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
                        : "=r"(_packed) : "f"(score_values_1[28]), "f"(score_values_1[29]),
                                           "f"(score_values_1[30]), "f"(score_values_1[31]));
                    packed_p_1[7] = _packed;
                }
                float _fp8_rt_64;
                uint16_t _e4m3x2_30;
                uint32_t _f16x2_30;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_30) : "f"(0.0f), "f"(score_values_1[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_30) : "h"(_e4m3x2_30));
                uint16_t _fp8_h0_30 = (uint16_t)(_f16x2_30 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_64) : "h"(_fp8_h0_30));
                rsum_1 = rsum_1 + _fp8_rt_64;
                float _fp8_rt_65;
                uint16_t _e4m3x2_31;
                uint32_t _f16x2_31;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_31) : "f"(0.0f), "f"(score_values_1[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_31) : "h"(_e4m3x2_31));
                uint16_t _fp8_h0_31 = (uint16_t)(_f16x2_31 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_65) : "h"(_fp8_h0_31));
                rsum_1 = rsum_1 + _fp8_rt_65;
                float _fp8_rt_66;
                uint16_t _e4m3x2_32;
                uint32_t _f16x2_32;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_32) : "f"(0.0f), "f"(score_values_1[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_32) : "h"(_e4m3x2_32));
                uint16_t _fp8_h0_32 = (uint16_t)(_f16x2_32 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_66) : "h"(_fp8_h0_32));
                rsum_1 = rsum_1 + _fp8_rt_66;
                float _fp8_rt_67;
                uint16_t _e4m3x2_33;
                uint32_t _f16x2_33;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_33) : "f"(0.0f), "f"(score_values_1[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_33) : "h"(_e4m3x2_33));
                uint16_t _fp8_h0_33 = (uint16_t)(_f16x2_33 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_67) : "h"(_fp8_h0_33));
                rsum_1 = rsum_1 + _fp8_rt_67;
                float _fp8_rt_68;
                uint16_t _e4m3x2_34;
                uint32_t _f16x2_34;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_34) : "f"(0.0f), "f"(score_values_1[4]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_34) : "h"(_e4m3x2_34));
                uint16_t _fp8_h0_34 = (uint16_t)(_f16x2_34 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_68) : "h"(_fp8_h0_34));
                rsum_1 = rsum_1 + _fp8_rt_68;
                float _fp8_rt_69;
                uint16_t _e4m3x2_35;
                uint32_t _f16x2_35;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_35) : "f"(0.0f), "f"(score_values_1[5]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_35) : "h"(_e4m3x2_35));
                uint16_t _fp8_h0_35 = (uint16_t)(_f16x2_35 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_69) : "h"(_fp8_h0_35));
                rsum_1 = rsum_1 + _fp8_rt_69;
                float _fp8_rt_70;
                uint16_t _e4m3x2_36;
                uint32_t _f16x2_36;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_36) : "f"(0.0f), "f"(score_values_1[6]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_36) : "h"(_e4m3x2_36));
                uint16_t _fp8_h0_36 = (uint16_t)(_f16x2_36 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_70) : "h"(_fp8_h0_36));
                rsum_1 = rsum_1 + _fp8_rt_70;
                float _fp8_rt_71;
                uint16_t _e4m3x2_37;
                uint32_t _f16x2_37;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_37) : "f"(0.0f), "f"(score_values_1[7]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_37) : "h"(_e4m3x2_37));
                uint16_t _fp8_h0_37 = (uint16_t)(_f16x2_37 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_71) : "h"(_fp8_h0_37));
                rsum_1 = rsum_1 + _fp8_rt_71;
                float _fp8_rt_72;
                uint16_t _e4m3x2_38;
                uint32_t _f16x2_38;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_38) : "f"(0.0f), "f"(score_values_1[8]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_38) : "h"(_e4m3x2_38));
                uint16_t _fp8_h0_38 = (uint16_t)(_f16x2_38 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_72) : "h"(_fp8_h0_38));
                rsum_1 = rsum_1 + _fp8_rt_72;
                float _fp8_rt_73;
                uint16_t _e4m3x2_39;
                uint32_t _f16x2_39;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_39) : "f"(0.0f), "f"(score_values_1[9]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_39) : "h"(_e4m3x2_39));
                uint16_t _fp8_h0_39 = (uint16_t)(_f16x2_39 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_73) : "h"(_fp8_h0_39));
                rsum_1 = rsum_1 + _fp8_rt_73;
                float _fp8_rt_74;
                uint16_t _e4m3x2_40;
                uint32_t _f16x2_40;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_40) : "f"(0.0f), "f"(score_values_1[10]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_40) : "h"(_e4m3x2_40));
                uint16_t _fp8_h0_40 = (uint16_t)(_f16x2_40 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_74) : "h"(_fp8_h0_40));
                rsum_1 = rsum_1 + _fp8_rt_74;
                float _fp8_rt_75;
                uint16_t _e4m3x2_41;
                uint32_t _f16x2_41;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_41) : "f"(0.0f), "f"(score_values_1[11]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_41) : "h"(_e4m3x2_41));
                uint16_t _fp8_h0_41 = (uint16_t)(_f16x2_41 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_75) : "h"(_fp8_h0_41));
                rsum_1 = rsum_1 + _fp8_rt_75;
                float _fp8_rt_76;
                uint16_t _e4m3x2_42;
                uint32_t _f16x2_42;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_42) : "f"(0.0f), "f"(score_values_1[12]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_42) : "h"(_e4m3x2_42));
                uint16_t _fp8_h0_42 = (uint16_t)(_f16x2_42 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_76) : "h"(_fp8_h0_42));
                rsum_1 = rsum_1 + _fp8_rt_76;
                float _fp8_rt_77;
                uint16_t _e4m3x2_43;
                uint32_t _f16x2_43;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_43) : "f"(0.0f), "f"(score_values_1[13]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_43) : "h"(_e4m3x2_43));
                uint16_t _fp8_h0_43 = (uint16_t)(_f16x2_43 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_77) : "h"(_fp8_h0_43));
                rsum_1 = rsum_1 + _fp8_rt_77;
                float _fp8_rt_78;
                uint16_t _e4m3x2_44;
                uint32_t _f16x2_44;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_44) : "f"(0.0f), "f"(score_values_1[14]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_44) : "h"(_e4m3x2_44));
                uint16_t _fp8_h0_44 = (uint16_t)(_f16x2_44 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_78) : "h"(_fp8_h0_44));
                rsum_1 = rsum_1 + _fp8_rt_78;
                float _fp8_rt_79;
                uint16_t _e4m3x2_45;
                uint32_t _f16x2_45;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_45) : "f"(0.0f), "f"(score_values_1[15]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_45) : "h"(_e4m3x2_45));
                uint16_t _fp8_h0_45 = (uint16_t)(_f16x2_45 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_79) : "h"(_fp8_h0_45));
                rsum_1 = rsum_1 + _fp8_rt_79;
                float _fp8_rt_80;
                uint16_t _e4m3x2_46;
                uint32_t _f16x2_46;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_46) : "f"(0.0f), "f"(score_values_1[16]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_46) : "h"(_e4m3x2_46));
                uint16_t _fp8_h0_46 = (uint16_t)(_f16x2_46 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_80) : "h"(_fp8_h0_46));
                rsum_1 = rsum_1 + _fp8_rt_80;
                float _fp8_rt_81;
                uint16_t _e4m3x2_47;
                uint32_t _f16x2_47;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_47) : "f"(0.0f), "f"(score_values_1[17]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_47) : "h"(_e4m3x2_47));
                uint16_t _fp8_h0_47 = (uint16_t)(_f16x2_47 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_81) : "h"(_fp8_h0_47));
                rsum_1 = rsum_1 + _fp8_rt_81;
                float _fp8_rt_82;
                uint16_t _e4m3x2_48;
                uint32_t _f16x2_48;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_48) : "f"(0.0f), "f"(score_values_1[18]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_48) : "h"(_e4m3x2_48));
                uint16_t _fp8_h0_48 = (uint16_t)(_f16x2_48 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_82) : "h"(_fp8_h0_48));
                rsum_1 = rsum_1 + _fp8_rt_82;
                float _fp8_rt_83;
                uint16_t _e4m3x2_49;
                uint32_t _f16x2_49;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_49) : "f"(0.0f), "f"(score_values_1[19]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_49) : "h"(_e4m3x2_49));
                uint16_t _fp8_h0_49 = (uint16_t)(_f16x2_49 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_83) : "h"(_fp8_h0_49));
                rsum_1 = rsum_1 + _fp8_rt_83;
                float _fp8_rt_84;
                uint16_t _e4m3x2_50;
                uint32_t _f16x2_50;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_50) : "f"(0.0f), "f"(score_values_1[20]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_50) : "h"(_e4m3x2_50));
                uint16_t _fp8_h0_50 = (uint16_t)(_f16x2_50 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_84) : "h"(_fp8_h0_50));
                rsum_1 = rsum_1 + _fp8_rt_84;
                float _fp8_rt_85;
                uint16_t _e4m3x2_51;
                uint32_t _f16x2_51;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_51) : "f"(0.0f), "f"(score_values_1[21]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_51) : "h"(_e4m3x2_51));
                uint16_t _fp8_h0_51 = (uint16_t)(_f16x2_51 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_85) : "h"(_fp8_h0_51));
                rsum_1 = rsum_1 + _fp8_rt_85;
                float _fp8_rt_86;
                uint16_t _e4m3x2_52;
                uint32_t _f16x2_52;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_52) : "f"(0.0f), "f"(score_values_1[22]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_52) : "h"(_e4m3x2_52));
                uint16_t _fp8_h0_52 = (uint16_t)(_f16x2_52 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_86) : "h"(_fp8_h0_52));
                rsum_1 = rsum_1 + _fp8_rt_86;
                float _fp8_rt_87;
                uint16_t _e4m3x2_53;
                uint32_t _f16x2_53;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_53) : "f"(0.0f), "f"(score_values_1[23]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_53) : "h"(_e4m3x2_53));
                uint16_t _fp8_h0_53 = (uint16_t)(_f16x2_53 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_87) : "h"(_fp8_h0_53));
                rsum_1 = rsum_1 + _fp8_rt_87;
                float _fp8_rt_88;
                uint16_t _e4m3x2_54;
                uint32_t _f16x2_54;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_54) : "f"(0.0f), "f"(score_values_1[24]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_54) : "h"(_e4m3x2_54));
                uint16_t _fp8_h0_54 = (uint16_t)(_f16x2_54 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_88) : "h"(_fp8_h0_54));
                rsum_1 = rsum_1 + _fp8_rt_88;
                float _fp8_rt_89;
                uint16_t _e4m3x2_55;
                uint32_t _f16x2_55;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_55) : "f"(0.0f), "f"(score_values_1[25]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_55) : "h"(_e4m3x2_55));
                uint16_t _fp8_h0_55 = (uint16_t)(_f16x2_55 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_89) : "h"(_fp8_h0_55));
                rsum_1 = rsum_1 + _fp8_rt_89;
                float _fp8_rt_90;
                uint16_t _e4m3x2_56;
                uint32_t _f16x2_56;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_56) : "f"(0.0f), "f"(score_values_1[26]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_56) : "h"(_e4m3x2_56));
                uint16_t _fp8_h0_56 = (uint16_t)(_f16x2_56 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_90) : "h"(_fp8_h0_56));
                rsum_1 = rsum_1 + _fp8_rt_90;
                float _fp8_rt_91;
                uint16_t _e4m3x2_57;
                uint32_t _f16x2_57;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_57) : "f"(0.0f), "f"(score_values_1[27]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_57) : "h"(_e4m3x2_57));
                uint16_t _fp8_h0_57 = (uint16_t)(_f16x2_57 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_91) : "h"(_fp8_h0_57));
                rsum_1 = rsum_1 + _fp8_rt_91;
                float _fp8_rt_92;
                uint16_t _e4m3x2_58;
                uint32_t _f16x2_58;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_58) : "f"(0.0f), "f"(score_values_1[28]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_58) : "h"(_e4m3x2_58));
                uint16_t _fp8_h0_58 = (uint16_t)(_f16x2_58 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_92) : "h"(_fp8_h0_58));
                rsum_1 = rsum_1 + _fp8_rt_92;
                float _fp8_rt_93;
                uint16_t _e4m3x2_59;
                uint32_t _f16x2_59;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_59) : "f"(0.0f), "f"(score_values_1[29]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_59) : "h"(_e4m3x2_59));
                uint16_t _fp8_h0_59 = (uint16_t)(_f16x2_59 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_93) : "h"(_fp8_h0_59));
                rsum_1 = rsum_1 + _fp8_rt_93;
                float _fp8_rt_94;
                uint16_t _e4m3x2_60;
                uint32_t _f16x2_60;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_60) : "f"(0.0f), "f"(score_values_1[30]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_60) : "h"(_e4m3x2_60));
                uint16_t _fp8_h0_60 = (uint16_t)(_f16x2_60 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_94) : "h"(_fp8_h0_60));
                rsum_1 = rsum_1 + _fp8_rt_94;
                float _fp8_rt_95;
                uint16_t _e4m3x2_61;
                uint32_t _f16x2_61;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_61) : "f"(0.0f), "f"(score_values_1[31]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_61) : "h"(_e4m3x2_61));
                uint16_t _fp8_h0_61 = (uint16_t)(_f16x2_61 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_95) : "h"(_fp8_h0_61));
                rsum_1 = rsum_1 + _fp8_rt_95;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_p_addr + (unsigned int)(row_1 * 128 + (64 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_p_addr + (unsigned int)(row_1 * 128 + (80 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_1[(4) + 3])));
            }
            float sink_term_1 = 0.0f;
            if (has_sink_row_1 != 0) {
                float _exp2_1 = approx_exp2(sink_log2_1 - row_max_scaled_1);
                sink_term_1 = _exp2_1;
            }
            smem_psum[128 + row_1] = slice_sum_1;
            smem_rsum[128 + row_1] = rsum_1;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(p_full_addr);
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            float denom_1 = smem_rsum[row_1] + smem_rsum[128 + row_1] + smem_rsum[256 + row_1] + sink_term_1;
            float _rcp_1 = approx_rcp(denom_1);
            float norm_1 = ((denom_1 > 0.0f) ? _rcp_1 * output_scale_1 : 0.0f);
            float o_values_1[64];
            unsigned int packed_1[32];
            long long out_base_1 = ((long long)(query_idx_1 * num_heads + head_row_1) * (long long)num_splits + (long long)split_idx_1) * 512;
            unsigned int _phase_o_full_1 = 0;
            if (num_heads <= 32) {
                mbarrier_wait_hint(o_full_addr + 8, _phase_o_full_1, 10000000);
                _phase_o_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp_rows_valid_1 != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values_1[0]), "=f"(o_values_1[1]), "=f"(o_values_1[2]), "=f"(o_values_1[3]), "=f"(o_values_1[4]), "=f"(o_values_1[5]), "=f"(o_values_1[6]), "=f"(o_values_1[7]), "=f"(o_values_1[8]), "=f"(o_values_1[9]), "=f"(o_values_1[10]), "=f"(o_values_1[11]), "=f"(o_values_1[12]), "=f"(o_values_1[13]), "=f"(o_values_1[14]), "=f"(o_values_1[15]), "=f"(o_values_1[16]), "=f"(o_values_1[17]), "=f"(o_values_1[18]), "=f"(o_values_1[19]), "=f"(o_values_1[20]), "=f"(o_values_1[21]), "=f"(o_values_1[22]), "=f"(o_values_1[23]), "=f"(o_values_1[24]), "=f"(o_values_1[25]), "=f"(o_values_1[26]), "=f"(o_values_1[27]), "=f"(o_values_1[28]), "=f"(o_values_1[29]), "=f"(o_values_1[30]), "=f"(o_values_1[31])
                        : "r"(taddr + 256 + (unsigned int)(tmem_row_origin_1 << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values_1[32]), "=f"(o_values_1[33]), "=f"(o_values_1[34]), "=f"(o_values_1[35]), "=f"(o_values_1[36]), "=f"(o_values_1[37]), "=f"(o_values_1[38]), "=f"(o_values_1[39]), "=f"(o_values_1[40]), "=f"(o_values_1[41]), "=f"(o_values_1[42]), "=f"(o_values_1[43]), "=f"(o_values_1[44]), "=f"(o_values_1[45]), "=f"(o_values_1[46]), "=f"(o_values_1[47]), "=f"(o_values_1[48]), "=f"(o_values_1[49]), "=f"(o_values_1[50]), "=f"(o_values_1[51]), "=f"(o_values_1[52]), "=f"(o_values_1[53]), "=f"(o_values_1[54]), "=f"(o_values_1[55]), "=f"(o_values_1[56]), "=f"(o_values_1[57]), "=f"(o_values_1[58]), "=f"(o_values_1[59]), "=f"(o_values_1[60]), "=f"(o_values_1[61]), "=f"(o_values_1[62]), "=f"(o_values_1[63])
                        : "r"(taddr + 256 + (unsigned int)(tmem_row_origin_1 << 16) + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_62 = {norm_1, norm_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_1)[_ls], _scale2_62);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++) {
                        o_values_1[_ls] = o_values_1[_ls] * norm_1;
                    }
                    #endif
                    if (row_valid_1 != 0) {
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values_1[0 + 0], o_values_1[0 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values_1[0 + 2], o_values_1[0 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values_1[0 + 4], o_values_1[0 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values_1[0 + 6], o_values_1[0 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values_1[0 + 8], o_values_1[0 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values_1[0 + 10], o_values_1[0 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values_1[0 + 12], o_values_1[0 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values_1[0 + 14], o_values_1[0 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base_1 + (long long)((o_chunk_1 * 2 + 1) * 128))))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values_1[16 + 0], o_values_1[16 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values_1[16 + 2], o_values_1[16 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values_1[16 + 4], o_values_1[16 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values_1[16 + 6], o_values_1[16 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values_1[16 + 8], o_values_1[16 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values_1[16 + 10], o_values_1[16 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values_1[16 + 12], o_values_1[16 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values_1[16 + 14], o_values_1[16 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base_1 + (long long)((o_chunk_1 * 2 + 1) * 128) + 16)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values_1[32 + 0], o_values_1[32 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values_1[32 + 2], o_values_1[32 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values_1[32 + 4], o_values_1[32 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values_1[32 + 6], o_values_1[32 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values_1[32 + 8], o_values_1[32 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values_1[32 + 10], o_values_1[32 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values_1[32 + 12], o_values_1[32 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values_1[32 + 14], o_values_1[32 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base_1 + (long long)((o_chunk_1 * 2 + 1) * 128) + 32)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values_1[48 + 0], o_values_1[48 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values_1[48 + 2], o_values_1[48 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values_1[48 + 4], o_values_1[48 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values_1[48 + 6], o_values_1[48 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values_1[48 + 8], o_values_1[48 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values_1[48 + 10], o_values_1[48 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values_1[48 + 12], o_values_1[48 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values_1[48 + 14], o_values_1[48 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base_1 + (long long)((o_chunk_1 * 2 + 1) * 128) + 48)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values_1[0]), "=f"(o_values_1[1]), "=f"(o_values_1[2]), "=f"(o_values_1[3]), "=f"(o_values_1[4]), "=f"(o_values_1[5]), "=f"(o_values_1[6]), "=f"(o_values_1[7]), "=f"(o_values_1[8]), "=f"(o_values_1[9]), "=f"(o_values_1[10]), "=f"(o_values_1[11]), "=f"(o_values_1[12]), "=f"(o_values_1[13]), "=f"(o_values_1[14]), "=f"(o_values_1[15]), "=f"(o_values_1[16]), "=f"(o_values_1[17]), "=f"(o_values_1[18]), "=f"(o_values_1[19]), "=f"(o_values_1[20]), "=f"(o_values_1[21]), "=f"(o_values_1[22]), "=f"(o_values_1[23]), "=f"(o_values_1[24]), "=f"(o_values_1[25]), "=f"(o_values_1[26]), "=f"(o_values_1[27]), "=f"(o_values_1[28]), "=f"(o_values_1[29]), "=f"(o_values_1[30]), "=f"(o_values_1[31])
                        : "r"(taddr + 256 + 64 + (unsigned int)(tmem_row_origin_1 << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values_1[32]), "=f"(o_values_1[33]), "=f"(o_values_1[34]), "=f"(o_values_1[35]), "=f"(o_values_1[36]), "=f"(o_values_1[37]), "=f"(o_values_1[38]), "=f"(o_values_1[39]), "=f"(o_values_1[40]), "=f"(o_values_1[41]), "=f"(o_values_1[42]), "=f"(o_values_1[43]), "=f"(o_values_1[44]), "=f"(o_values_1[45]), "=f"(o_values_1[46]), "=f"(o_values_1[47]), "=f"(o_values_1[48]), "=f"(o_values_1[49]), "=f"(o_values_1[50]), "=f"(o_values_1[51]), "=f"(o_values_1[52]), "=f"(o_values_1[53]), "=f"(o_values_1[54]), "=f"(o_values_1[55]), "=f"(o_values_1[56]), "=f"(o_values_1[57]), "=f"(o_values_1[58]), "=f"(o_values_1[59]), "=f"(o_values_1[60]), "=f"(o_values_1[61]), "=f"(o_values_1[62]), "=f"(o_values_1[63])
                        : "r"(taddr + 256 + 64 + (unsigned int)(tmem_row_origin_1 << 16) + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_63 = {norm_1, norm_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_1)[_ls], _scale2_63);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++) {
                        o_values_1[_ls] = o_values_1[_ls] * norm_1;
                    }
                    #endif
                    if (row_valid_1 != 0) {
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values_1[0 + 0], o_values_1[0 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values_1[0 + 2], o_values_1[0 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values_1[0 + 4], o_values_1[0 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values_1[0 + 6], o_values_1[0 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values_1[0 + 8], o_values_1[0 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values_1[0 + 10], o_values_1[0 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values_1[0 + 12], o_values_1[0 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values_1[0 + 14], o_values_1[0 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base_1 + (long long)((o_chunk_1 * 2 + 1) * 128) + 64)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values_1[16 + 0], o_values_1[16 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values_1[16 + 2], o_values_1[16 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values_1[16 + 4], o_values_1[16 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values_1[16 + 6], o_values_1[16 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values_1[16 + 8], o_values_1[16 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values_1[16 + 10], o_values_1[16 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values_1[16 + 12], o_values_1[16 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values_1[16 + 14], o_values_1[16 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base_1 + (long long)((o_chunk_1 * 2 + 1) * 128) + 64 + 16)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values_1[32 + 0], o_values_1[32 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values_1[32 + 2], o_values_1[32 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values_1[32 + 4], o_values_1[32 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values_1[32 + 6], o_values_1[32 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values_1[32 + 8], o_values_1[32 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values_1[32 + 10], o_values_1[32 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values_1[32 + 12], o_values_1[32 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values_1[32 + 14], o_values_1[32 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base_1 + (long long)((o_chunk_1 * 2 + 1) * 128) + 64 + 32)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(o_values_1[48 + 0], o_values_1[48 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(o_values_1[48 + 2], o_values_1[48 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(o_values_1[48 + 4], o_values_1[48 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(o_values_1[48 + 6], o_values_1[48 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(o_values_1[48 + 8], o_values_1[48 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(o_values_1[48 + 10], o_values_1[48 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(o_values_1[48 + 12], o_values_1[48 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(o_values_1[48 + 14], o_values_1[48 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(partial_O + (out_base_1 + (long long)((o_chunk_1 * 2 + 1) * 128) + 64 + 48)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                    }
                }
            } else {
                mbarrier_wait_hint(o_full_addr + 8, _phase_o_full_1, 10000000);
                _phase_o_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp_rows_valid_1 != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values_1[0]), "=f"(o_values_1[1]), "=f"(o_values_1[2]), "=f"(o_values_1[3]), "=f"(o_values_1[4]), "=f"(o_values_1[5]), "=f"(o_values_1[6]), "=f"(o_values_1[7]), "=f"(o_values_1[8]), "=f"(o_values_1[9]), "=f"(o_values_1[10]), "=f"(o_values_1[11]), "=f"(o_values_1[12]), "=f"(o_values_1[13]), "=f"(o_values_1[14]), "=f"(o_values_1[15]), "=f"(o_values_1[16]), "=f"(o_values_1[17]), "=f"(o_values_1[18]), "=f"(o_values_1[19]), "=f"(o_values_1[20]), "=f"(o_values_1[21]), "=f"(o_values_1[22]), "=f"(o_values_1[23]), "=f"(o_values_1[24]), "=f"(o_values_1[25]), "=f"(o_values_1[26]), "=f"(o_values_1[27]), "=f"(o_values_1[28]), "=f"(o_values_1[29]), "=f"(o_values_1[30]), "=f"(o_values_1[31])
                        : "r"(taddr + 256 + (unsigned int)(tmem_row_origin_1 << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values_1[32]), "=f"(o_values_1[33]), "=f"(o_values_1[34]), "=f"(o_values_1[35]), "=f"(o_values_1[36]), "=f"(o_values_1[37]), "=f"(o_values_1[38]), "=f"(o_values_1[39]), "=f"(o_values_1[40]), "=f"(o_values_1[41]), "=f"(o_values_1[42]), "=f"(o_values_1[43]), "=f"(o_values_1[44]), "=f"(o_values_1[45]), "=f"(o_values_1[46]), "=f"(o_values_1[47]), "=f"(o_values_1[48]), "=f"(o_values_1[49]), "=f"(o_values_1[50]), "=f"(o_values_1[51]), "=f"(o_values_1[52]), "=f"(o_values_1[53]), "=f"(o_values_1[54]), "=f"(o_values_1[55]), "=f"(o_values_1[56]), "=f"(o_values_1[57]), "=f"(o_values_1[58]), "=f"(o_values_1[59]), "=f"(o_values_1[60]), "=f"(o_values_1[61]), "=f"(o_values_1[62]), "=f"(o_values_1[63])
                        : "r"(taddr + 256 + (unsigned int)(tmem_row_origin_1 << 16) + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_64 = {norm_1, norm_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_1)[_ls], _scale2_64);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++) {
                        o_values_1[_ls] = o_values_1[_ls] * norm_1;
                    }
                    #endif
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_1[_lp*2 + 0], o_values_1[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    int o_row_addr_1 = smem_ostage_addr + 32768 + (unsigned int)(row_1 * 128);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_1 + (0 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_1 + (1 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(4) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_1 + (2 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[8])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(8) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_1 + (3 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[12])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(12) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_1 + (4 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[16])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(16) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(16) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(16) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_1 + (5 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[20])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(20) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(20) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(20) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_1 + (6 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[24])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(24) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(24) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(24) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_1 + (7 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[28])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(28) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(28) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(28) + 3])));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values_1[0]), "=f"(o_values_1[1]), "=f"(o_values_1[2]), "=f"(o_values_1[3]), "=f"(o_values_1[4]), "=f"(o_values_1[5]), "=f"(o_values_1[6]), "=f"(o_values_1[7]), "=f"(o_values_1[8]), "=f"(o_values_1[9]), "=f"(o_values_1[10]), "=f"(o_values_1[11]), "=f"(o_values_1[12]), "=f"(o_values_1[13]), "=f"(o_values_1[14]), "=f"(o_values_1[15]), "=f"(o_values_1[16]), "=f"(o_values_1[17]), "=f"(o_values_1[18]), "=f"(o_values_1[19]), "=f"(o_values_1[20]), "=f"(o_values_1[21]), "=f"(o_values_1[22]), "=f"(o_values_1[23]), "=f"(o_values_1[24]), "=f"(o_values_1[25]), "=f"(o_values_1[26]), "=f"(o_values_1[27]), "=f"(o_values_1[28]), "=f"(o_values_1[29]), "=f"(o_values_1[30]), "=f"(o_values_1[31])
                        : "r"(taddr + 256 + 64 + (unsigned int)(tmem_row_origin_1 << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_values_1[32]), "=f"(o_values_1[33]), "=f"(o_values_1[34]), "=f"(o_values_1[35]), "=f"(o_values_1[36]), "=f"(o_values_1[37]), "=f"(o_values_1[38]), "=f"(o_values_1[39]), "=f"(o_values_1[40]), "=f"(o_values_1[41]), "=f"(o_values_1[42]), "=f"(o_values_1[43]), "=f"(o_values_1[44]), "=f"(o_values_1[45]), "=f"(o_values_1[46]), "=f"(o_values_1[47]), "=f"(o_values_1[48]), "=f"(o_values_1[49]), "=f"(o_values_1[50]), "=f"(o_values_1[51]), "=f"(o_values_1[52]), "=f"(o_values_1[53]), "=f"(o_values_1[54]), "=f"(o_values_1[55]), "=f"(o_values_1[56]), "=f"(o_values_1[57]), "=f"(o_values_1[58]), "=f"(o_values_1[59]), "=f"(o_values_1[60]), "=f"(o_values_1[61]), "=f"(o_values_1[62]), "=f"(o_values_1[63])
                        : "r"(taddr + 256 + 64 + (unsigned int)(tmem_row_origin_1 << 16) + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_65 = {norm_1, norm_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(o_values_1)[_ls], _scale2_65);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++) {
                        o_values_1[_ls] = o_values_1[_ls] * norm_1;
                    }
                    #endif
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_values_1[_lp*2 + 0], o_values_1[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    int o_row_addr_0_1 = smem_ostage_addr + 49152 + (unsigned int)(row_1 * 128);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0_1 + (0 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0_1 + (1 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(4) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0_1 + (2 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[8])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(8) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0_1 + (3 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[12])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(12) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0_1 + (4 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[16])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(16) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(16) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(16) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0_1 + (5 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[20])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(20) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(20) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(20) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0_1 + (6 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[24])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(24) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(24) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(24) + 3])));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(o_row_addr_0_1 + (7 ^ row_1 % 8) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_1[28])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(28) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(28) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_1[(28) + 3])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 11, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        tma_store_4d((&tmap_out), (o_chunk_1 * 2 + 1) * 128, split_idx_1, head_base_1, query_idx_1, smem_ostage_addr + 32768);
                        tma_store_4d((&tmap_out), (o_chunk_1 * 2 + 1) * 128 + 64, split_idx_1, head_base_1, query_idx_1, smem_ostage_addr + 49152);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                if (warp == 4) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: compute2 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // compute2_main
            const int local_warp_2 = warp - 8;
            int o_chunk_2 = blockIdx.x % 2;
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
            int is_main_2 = 1;
            if (split_idx_2 >= num_main_tiles) {
                is_main_2 = 0;
            }
            int tile_in_table_2 = ((is_main_2 != 0) ? split_idx_2 : split_idx_2 - num_main_tiles);
            int table_width_2 = ((is_main_2 != 0) ? main_width : extra_width);
            int* row_ptr_2 = ((is_main_2 != 0) ? (main_indices + (query_idx_2 * main_index_stride)) : (extra_indices + (query_idx_2 * extra_index_stride)));
            int col_2 = tile_in_table_2 * 128 + row_2;
            int raw_index_2 = -1;
            if (col_2 < table_width_2) {
                raw_index_2 = row_ptr_2[col_2];
            }
            int active_len_2 = table_width_2;
            if (is_main_2 != 0) {
                if (has_main_lengths != 0) {
                    active_len_2 = main_lengths[query_idx_2];
                }
            } else if (has_extra_lengths != 0) {
                active_len_2 = extra_lengths[query_idx_2];
            }
            if (active_len_2 < 0) {
                active_len_2 = 0;
            }
            if (active_len_2 > table_width_2) {
                active_len_2 = table_width_2;
            }
            int valid_2 = 1;
            if (raw_index_2 < 0) {
                valid_2 = 0;
            }
            if (col_2 >= active_len_2) {
                valid_2 = 0;
            }
            uint8_t* cache_2 = ((is_main_2 != 0) ? (main_cache) : (extra_cache));
            int page_shift_2 = ((is_main_2 != 0) ? main_page_shift : extra_page_shift);
            long long page_stride_2 = ((is_main_2 != 0) ? main_page_stride : extra_page_stride);
            int safe_index_2 = ((raw_index_2 >= 0) ? raw_index_2 : 0);
            int page_2 = safe_index_2 >> page_shift_2;
            int slot_in_page_2 = safe_index_2 - (page_2 << page_shift_2);
            int page_size_2 = 1 << page_shift_2;
            long long page_base_2 = (long long)page_2 * page_stride_2;
            long long data_off_2 = page_base_2 + (long long)(slot_in_page_2 * 352);
            long long sf_off_2 = page_base_2 + (long long)(page_size_2 * 352 + slot_in_page_2 * 32);
            int slot_2 = smem_raw_addr + (unsigned int)(row_2 * 400);
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            const int gather_tid_2 = warp * 32 + lane;
            const int g_chunk_2 = gather_tid_2 % 24;
            const int g_row0_2 = gather_tid_2 / 24;
            const int g_is_sf_2 = ((g_chunk_2 >= 22) ? 1 : 0);
            long long g_src_off_2 = (long long)(((g_is_sf_2 != 0) ? 16 * (g_chunk_2 - 14 - 8) : 16 * g_chunk_2));
            int g_dst0_2 = smem_raw_addr + (unsigned int)(g_row0_2 * 400) + (unsigned int)(16 * g_chunk_2);
            unsigned int w4_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)(g_row0_2 * 16)));
            long long od_3 = (long long)w4_2[1] << 32 | (long long)w4_2[0];
            long long osf_3 = (long long)w4_2[3] << 32 | (long long)w4_2[2];
            long long goff_2 = ((g_is_sf_2 != 0) ? osf_3 : od_3);
            if (goff_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2), "l"(cache_2 + (goff_2 + g_src_off_2)));
            }
            unsigned int w4_0_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 16) * 16)));
            long long od_1_2 = (long long)w4_0_2[1] << 32 | (long long)w4_0_2[0];
            long long osf_2_2 = (long long)w4_0_2[3] << 32 | (long long)w4_0_2[2];
            long long goff_3_2 = ((g_is_sf_2 != 0) ? osf_2_2 : od_1_2);
            if (goff_3_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 6400), "l"(cache_2 + (goff_3_2 + g_src_off_2)));
            }
            unsigned int w4_4_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 32) * 16)));
            long long od_5_2 = (long long)w4_4_2[1] << 32 | (long long)w4_4_2[0];
            long long osf_6_2 = (long long)w4_4_2[3] << 32 | (long long)w4_4_2[2];
            long long goff_7_2 = ((g_is_sf_2 != 0) ? osf_6_2 : od_5_2);
            if (goff_7_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 12800), "l"(cache_2 + (goff_7_2 + g_src_off_2)));
            }
            unsigned int w4_8_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_8_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 48) * 16)));
            long long od_9_2 = (long long)w4_8_2[1] << 32 | (long long)w4_8_2[0];
            long long osf_10_2 = (long long)w4_8_2[3] << 32 | (long long)w4_8_2[2];
            long long goff_11_2 = ((g_is_sf_2 != 0) ? osf_10_2 : od_9_2);
            if (goff_11_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 19200), "l"(cache_2 + (goff_11_2 + g_src_off_2)));
            }
            unsigned int w4_12_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 64) * 16)));
            long long od_13_2 = (long long)w4_12_2[1] << 32 | (long long)w4_12_2[0];
            long long osf_14_2 = (long long)w4_12_2[3] << 32 | (long long)w4_12_2[2];
            long long goff_15_2 = ((g_is_sf_2 != 0) ? osf_14_2 : od_13_2);
            if (goff_15_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 25600), "l"(cache_2 + (goff_15_2 + g_src_off_2)));
            }
            unsigned int w4_16_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_16_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 80) * 16)));
            long long od_17_2 = (long long)w4_16_2[1] << 32 | (long long)w4_16_2[0];
            long long osf_18_2 = (long long)w4_16_2[3] << 32 | (long long)w4_16_2[2];
            long long goff_19_2 = ((g_is_sf_2 != 0) ? osf_18_2 : od_17_2);
            if (goff_19_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 32000), "l"(cache_2 + (goff_19_2 + g_src_off_2)));
            }
            unsigned int w4_20_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_20_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 96) * 16)));
            long long od_21_2 = (long long)w4_20_2[1] << 32 | (long long)w4_20_2[0];
            long long osf_22_2 = (long long)w4_20_2[3] << 32 | (long long)w4_20_2[2];
            long long goff_23_2 = ((g_is_sf_2 != 0) ? osf_22_2 : od_21_2);
            if (goff_23_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 38400), "l"(cache_2 + (goff_23_2 + g_src_off_2)));
            }
            unsigned int w4_24_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_24_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 112) * 16)));
            long long od_25_2 = (long long)w4_24_2[1] << 32 | (long long)w4_24_2[0];
            long long osf_26_2 = (long long)w4_24_2[3] << 32 | (long long)w4_24_2[2];
            long long goff_27_2 = ((g_is_sf_2 != 0) ? osf_26_2 : od_25_2);
            if (goff_27_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 44800), "l"(cache_2 + (goff_27_2 + g_src_off_2)));
            }
            asm volatile("cp.async.commit_group;");
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
            for (int i_2 = 0; i_2 < 3; i_2++) {
                int unit_2 = q_warp_2 + 12 * i_2;
                if (unit_2 < 28) {
                    int q_block_2 = unit_2 / 7;
                    int kset_2 = unit_2 - q_block_2 * 7;
                    int q_row_2 = q_block_2 * 32 + lane;
                    if (head_base_2 + q_block_2 * 32 < num_heads) {
                        int q_row_addr_2 = smem_qstage_addr + (unsigned int)(kset_2 * 16384) + (unsigned int)(q_row_2 * 128);
                        unsigned int sf_word_4 = 0;
                        for (int bp_2 = 0; bp_2 < 2; bp_2++) {
                            unsigned int words_2[4];
                            unsigned int qa_2[4];
                            unsigned int qb_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_2[(0) + 3]))
                                : "r"(q_row_addr_2 + (4 * bp_2 ^ q_row_2 % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_3[(0) + 3]))
                                : "r"(q_row_addr_2 + (4 * bp_2 + 1 ^ q_row_2 % 8) * 16));
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
                            float _fabs_64 = fabsf(qv_3[0]);
                            float _fabs_65 = fabsf(qv_3[1]);
                            float _max_66 = max_noftz(_fabs_64, _fabs_65);
                            m8_2[0] = _max_66;
                            float _fabs_66 = fabsf(qv_3[2]);
                            float _fabs_67 = fabsf(qv_3[3]);
                            float _max_67 = max_noftz(_fabs_66, _fabs_67);
                            m8_2[1] = _max_67;
                            float _fabs_68 = fabsf(qv_3[4]);
                            float _fabs_69 = fabsf(qv_3[5]);
                            float _max_68 = max_noftz(_fabs_68, _fabs_69);
                            m8_2[2] = _max_68;
                            float _fabs_70 = fabsf(qv_3[6]);
                            float _fabs_71 = fabsf(qv_3[7]);
                            float _max_69 = max_noftz(_fabs_70, _fabs_71);
                            m8_2[3] = _max_69;
                            float _fabs_72 = fabsf(qv_3[8]);
                            float _fabs_73 = fabsf(qv_3[9]);
                            float _max_70 = max_noftz(_fabs_72, _fabs_73);
                            m8_2[4] = _max_70;
                            float _fabs_74 = fabsf(qv_3[10]);
                            float _fabs_75 = fabsf(qv_3[11]);
                            float _max_71 = max_noftz(_fabs_74, _fabs_75);
                            m8_2[5] = _max_71;
                            float _fabs_76 = fabsf(qv_3[12]);
                            float _fabs_77 = fabsf(qv_3[13]);
                            float _max_72 = max_noftz(_fabs_76, _fabs_77);
                            m8_2[6] = _max_72;
                            float _fabs_78 = fabsf(qv_3[14]);
                            float _fabs_79 = fabsf(qv_3[15]);
                            float _max_73 = max_noftz(_fabs_78, _fabs_79);
                            m8_2[7] = _max_73;
                            float m4_2[4];
                            float _max_74 = max_noftz(m8_2[0], m8_2[1]);
                            m4_2[0] = _max_74;
                            float _max_75 = max_noftz(m8_2[2], m8_2[3]);
                            m4_2[1] = _max_75;
                            float _max_76 = max_noftz(m8_2[4], m8_2[5]);
                            m4_2[2] = _max_76;
                            float _max_77 = max_noftz(m8_2[6], m8_2[7]);
                            m4_2[3] = _max_77;
                            float _max_78 = max_noftz(m4_2[0], m4_2[1]);
                            float _max_79 = max_noftz(m4_2[2], m4_2[3]);
                            float _max_80 = max_noftz(_max_78, _max_79);
                            float amax_2 = _max_80;
                            float sc_2 = amax_2 * inv_six_2;
                            uint16_t _e4m3x2_f32_20;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_20) : "f"(0.0f), "f"(sc_2));
                            uint16_t sc_pair_2 = _e4m3x2_f32_20;
                            unsigned int sc_byte_2 = (unsigned int)sc_pair_2 & 255;
                            unsigned int sc_exp_2 = sc_byte_2 >> 3 & 15;
                            unsigned int sc_man_2 = sc_byte_2 & 7;
                            float inv_2 = 0.0f;
                            if (sc_exp_2 == 0) {
                                inv_2 = __uint_as_float(smem_rcptab[8 + sc_man_2]) * 512.0f;
                            } else {
                                inv_2 = __uint_as_float(smem_rcptab[sc_man_2]) * __uint_as_float(134 - sc_exp_2 << 23);
                            }
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
                            uint32_t _fp4_pair_32;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_32) : "f"(qv_3[0]), "f"(qv_3[1]));
                            uint32_t _fp4_pair_33;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_33) : "f"(qv_3[2]), "f"(qv_3[3]));
                            uint32_t _fp4_pair_34;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_34) : "f"(qv_3[4]), "f"(qv_3[5]));
                            uint32_t _fp4_pair_35;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_35) : "f"(qv_3[6]), "f"(qv_3[7]));
                            uint32_t _fp4_pair_36;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_36) : "f"(qv_3[8]), "f"(qv_3[9]));
                            uint32_t _fp4_pair_37;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_37) : "f"(qv_3[10]), "f"(qv_3[11]));
                            uint32_t _fp4_pair_38;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_38) : "f"(qv_3[12]), "f"(qv_3[13]));
                            uint32_t _fp4_pair_39;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_39) : "f"(qv_3[14]), "f"(qv_3[15]));
                            words_2[0] = _fp4_pair_32 | _fp4_pair_33 << 8 | _fp4_pair_34 << 16 | _fp4_pair_35 << 24;
                            words_2[1] = _fp4_pair_36 | _fp4_pair_37 << 8 | _fp4_pair_38 << 16 | _fp4_pair_39 << 24;
                            sf_word_4 = sf_word_4 | sc_byte_2 << (unsigned int)(8 * (2 * bp_2));
                            unsigned int qa_0_2[4];
                            unsigned int qb_1_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qa_0_2[(0) + 3]))
                                : "r"(q_row_addr_2 + (4 * bp_2 + 2 ^ q_row_2 % 8) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qb_1_2[(0) + 3]))
                                : "r"(q_row_addr_2 + (4 * bp_2 + 2 + 1 ^ q_row_2 % 8) * 16));
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
                            float _fabs_80 = fabsf(qv_2_2[0]);
                            float _fabs_81 = fabsf(qv_2_2[1]);
                            float _max_81 = max_noftz(_fabs_80, _fabs_81);
                            m8_3_2[0] = _max_81;
                            float _fabs_82 = fabsf(qv_2_2[2]);
                            float _fabs_83 = fabsf(qv_2_2[3]);
                            float _max_82 = max_noftz(_fabs_82, _fabs_83);
                            m8_3_2[1] = _max_82;
                            float _fabs_84 = fabsf(qv_2_2[4]);
                            float _fabs_85 = fabsf(qv_2_2[5]);
                            float _max_83 = max_noftz(_fabs_84, _fabs_85);
                            m8_3_2[2] = _max_83;
                            float _fabs_86 = fabsf(qv_2_2[6]);
                            float _fabs_87 = fabsf(qv_2_2[7]);
                            float _max_84 = max_noftz(_fabs_86, _fabs_87);
                            m8_3_2[3] = _max_84;
                            float _fabs_88 = fabsf(qv_2_2[8]);
                            float _fabs_89 = fabsf(qv_2_2[9]);
                            float _max_85 = max_noftz(_fabs_88, _fabs_89);
                            m8_3_2[4] = _max_85;
                            float _fabs_90 = fabsf(qv_2_2[10]);
                            float _fabs_91 = fabsf(qv_2_2[11]);
                            float _max_86 = max_noftz(_fabs_90, _fabs_91);
                            m8_3_2[5] = _max_86;
                            float _fabs_92 = fabsf(qv_2_2[12]);
                            float _fabs_93 = fabsf(qv_2_2[13]);
                            float _max_87 = max_noftz(_fabs_92, _fabs_93);
                            m8_3_2[6] = _max_87;
                            float _fabs_94 = fabsf(qv_2_2[14]);
                            float _fabs_95 = fabsf(qv_2_2[15]);
                            float _max_88 = max_noftz(_fabs_94, _fabs_95);
                            m8_3_2[7] = _max_88;
                            float m4_4_2[4];
                            float _max_89 = max_noftz(m8_3_2[0], m8_3_2[1]);
                            m4_4_2[0] = _max_89;
                            float _max_90 = max_noftz(m8_3_2[2], m8_3_2[3]);
                            m4_4_2[1] = _max_90;
                            float _max_91 = max_noftz(m8_3_2[4], m8_3_2[5]);
                            m4_4_2[2] = _max_91;
                            float _max_92 = max_noftz(m8_3_2[6], m8_3_2[7]);
                            m4_4_2[3] = _max_92;
                            float _max_93 = max_noftz(m4_4_2[0], m4_4_2[1]);
                            float _max_94 = max_noftz(m4_4_2[2], m4_4_2[3]);
                            float _max_95 = max_noftz(_max_93, _max_94);
                            float amax_5_2 = _max_95;
                            float sc_6_2 = amax_5_2 * inv_six_2;
                            uint16_t _e4m3x2_f32_21;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_21) : "f"(0.0f), "f"(sc_6_2));
                            uint16_t sc_pair_7_2 = _e4m3x2_f32_21;
                            unsigned int sc_byte_8_2 = (unsigned int)sc_pair_7_2 & 255;
                            unsigned int sc_exp_9_2 = sc_byte_8_2 >> 3 & 15;
                            unsigned int sc_man_10_2 = sc_byte_8_2 & 7;
                            float inv_11_2 = 0.0f;
                            if (sc_exp_9_2 == 0) {
                                inv_11_2 = __uint_as_float(smem_rcptab[8 + sc_man_10_2]) * 512.0f;
                            } else {
                                inv_11_2 = __uint_as_float(smem_rcptab[sc_man_10_2]) * __uint_as_float(134 - sc_exp_9_2 << 23);
                            }
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_1 = {inv_11_2, inv_11_2};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(qv_2_2)[_ls], _scale2_1);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                qv_2_2[_ls] = qv_2_2[_ls] * inv_11_2;
                            }
                            #endif
                            uint32_t _fp4_pair_40;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_40) : "f"(qv_2_2[0]), "f"(qv_2_2[1]));
                            uint32_t _fp4_pair_41;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_41) : "f"(qv_2_2[2]), "f"(qv_2_2[3]));
                            uint32_t _fp4_pair_42;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_42) : "f"(qv_2_2[4]), "f"(qv_2_2[5]));
                            uint32_t _fp4_pair_43;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_43) : "f"(qv_2_2[6]), "f"(qv_2_2[7]));
                            uint32_t _fp4_pair_44;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_44) : "f"(qv_2_2[8]), "f"(qv_2_2[9]));
                            uint32_t _fp4_pair_45;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_45) : "f"(qv_2_2[10]), "f"(qv_2_2[11]));
                            uint32_t _fp4_pair_46;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_46) : "f"(qv_2_2[12]), "f"(qv_2_2[13]));
                            uint32_t _fp4_pair_47;
                            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_47) : "f"(qv_2_2[14]), "f"(qv_2_2[15]));
                            words_2[2] = _fp4_pair_40 | _fp4_pair_41 << 8 | _fp4_pair_42 << 16 | _fp4_pair_43 << 24;
                            words_2[3] = _fp4_pair_44 | _fp4_pair_45 << 8 | _fp4_pair_46 << 16 | _fp4_pair_47 << 24;
                            sf_word_4 = sf_word_4 | sc_byte_8_2 << (unsigned int)(8 * (2 * bp_2 + 1));
                            int chunk_2 = 2 * kset_2 + bp_2;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_qf4_addr + (unsigned int)(chunk_2 / 8 * 16384 + (q_row_2 * 128 + (chunk_2 % 8 * 16 ^ q_row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words_2[0])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 3])));
                        }
                        smem_qsf32[kset_2 / 4 * 2048 + q_row_2 % 32 / 8 * 512 + kset_2 % 4 * 128 + q_row_2 % 8 * 16 + q_row_2 / 32 % 4 * 4 >> 2] = sf_word_4;
                    } else {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(2 * kset_2 / 8 * 16384 + (q_row_2 * 128 + (2 * kset_2 % 8 * 16 ^ q_row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)((2 * kset_2 + 1) / 8 * 16384 + (q_row_2 * 128 + ((2 * kset_2 + 1) % 8 * 16 ^ q_row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        smem_qsf32[kset_2 / 4 * 2048 + q_row_2 % 32 / 8 * 512 + kset_2 % 4 * 128 + q_row_2 % 8 * 16 + q_row_2 / 32 % 4 * 4 >> 2] = 0;
                    }
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
            asm volatile("cp.async.wait_group 0;");
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            int nope_hi_v_2 = 14;
            int rope_pairs_hi_v_2 = 4;
            if (valid_2 != 0) {
                {
                    for (int jp_2 = 1; jp_2 < rope_pairs_hi_v_2; jp_2++) {
                        unsigned int vrope_1[4];
                        int j_1 = 2 * jp_2;
                        unsigned int rope_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 3]))
                            : "r"(slot_2 + 224 + 16 * j_1));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + (j_1 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&rope_1[0])), "r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 3])));
                        float lo_1 = __uint_as_float(rope_1[0] << 16);
                        float hi_2 = __uint_as_float(rope_1[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_22;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_22) : "f"(hi_2), "f"(lo_1));
                        uint16_t pair_1 = _e4m3x2_f32_22;
                        {
                            vrope_1[0] = (unsigned int)pair_1;
                        }
                        float lo_0_1 = __uint_as_float(rope_1[1] << 16);
                        float hi_1_1 = __uint_as_float(rope_1[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_23;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_23) : "f"(hi_1_1), "f"(lo_0_1));
                        uint16_t pair_2_1 = _e4m3x2_f32_23;
                        {
                            vrope_1[0] = vrope_1[0] | (unsigned int)pair_2_1 << 16;
                        }
                        float lo_3_1 = __uint_as_float(rope_1[2] << 16);
                        float hi_4_1 = __uint_as_float(rope_1[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_24;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_24) : "f"(hi_4_1), "f"(lo_3_1));
                        uint16_t pair_5_1 = _e4m3x2_f32_24;
                        {
                            vrope_1[1] = (unsigned int)pair_5_1;
                        }
                        float lo_6_1 = __uint_as_float(rope_1[3] << 16);
                        float hi_7_1 = __uint_as_float(rope_1[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_25;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_25) : "f"(hi_7_1), "f"(lo_6_1));
                        uint16_t pair_8_1 = _e4m3x2_f32_25;
                        {
                            vrope_1[1] = vrope_1[1] | (unsigned int)pair_8_1 << 16;
                        }
                        int j_9_1 = 2 * jp_2 + 1;
                        unsigned int rope_10_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rope_10_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_10_1[(0) + 3]))
                            : "r"(slot_2 + 224 + 16 * j_9_1));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + (j_9_1 * 16 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&rope_10_1[0])), "r"(*reinterpret_cast<uint32_t*>(&rope_10_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&rope_10_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&rope_10_1[(0) + 3])));
                        float lo_11_1 = __uint_as_float(rope_10_1[0] << 16);
                        float hi_12_1 = __uint_as_float(rope_10_1[0] & 4294901760u);
                        uint16_t _e4m3x2_f32_26;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_26) : "f"(hi_12_1), "f"(lo_11_1));
                        uint16_t pair_13_1 = _e4m3x2_f32_26;
                        {
                            vrope_1[2] = (unsigned int)pair_13_1;
                        }
                        float lo_14_1 = __uint_as_float(rope_10_1[1] << 16);
                        float hi_15_1 = __uint_as_float(rope_10_1[1] & 4294901760u);
                        uint16_t _e4m3x2_f32_27;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_27) : "f"(hi_15_1), "f"(lo_14_1));
                        uint16_t pair_16_1 = _e4m3x2_f32_27;
                        {
                            vrope_1[2] = vrope_1[2] | (unsigned int)pair_16_1 << 16;
                        }
                        float lo_17_1 = __uint_as_float(rope_10_1[2] << 16);
                        float hi_18_1 = __uint_as_float(rope_10_1[2] & 4294901760u);
                        uint16_t _e4m3x2_f32_28;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_28) : "f"(hi_18_1), "f"(lo_17_1));
                        uint16_t pair_19_1 = _e4m3x2_f32_28;
                        {
                            vrope_1[3] = (unsigned int)pair_19_1;
                        }
                        float lo_20_1 = __uint_as_float(rope_10_1[3] << 16);
                        float hi_21_1 = __uint_as_float(rope_10_1[3] & 4294901760u);
                        uint16_t _e4m3x2_f32_29;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_29) : "f"(hi_21_1), "f"(lo_20_1));
                        uint16_t pair_22_1 = _e4m3x2_f32_29;
                        {
                            vrope_1[3] = vrope_1[3] | (unsigned int)pair_22_1 << 16;
                        }
                        if (o_chunk_2 == 1) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_v_addr + (unsigned int)((jp_2 - 4 + 16) / 8 * 16384 + (row_2 * 128 + ((jp_2 - 4 + 16) % 8 * 16 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&vrope_1[0])), "r"(*reinterpret_cast<uint32_t*>(&vrope_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&vrope_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&vrope_1[(0) + 3])));
                        }
                    }
                }
            } else {
                {
                    for (int jp_3 = 1; jp_3 < rope_pairs_hi_v_2; jp_3++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + (2 * jp_3 * 16 ^ row_2 % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + ((2 * jp_3 + 1) * 16 ^ row_2 % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        if (o_chunk_2 == 1) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)((jp_3 - 4 + 16) / 8 * 16384 + (row_2 * 128 + ((jp_3 - 4 + 16) % 8 * 16 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        }
                    }
                }
            }
            {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kf4_addr + (unsigned int)(16384 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kf4_addr + (unsigned int)(16384 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            float softmax_scale_log2_2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_2 = bmm2_scale[0];
            float sink_log2_2 = 0.0f;
            int has_sink_row_2 = 0;
            if (has_sinks != 0 && split_idx_2 == 0 && row_valid_2 != 0) {
                has_sink_row_2 = 1;
                sink_log2_2 = sinks[head_row_2] * 1.4426950408889634f;
            }
            unsigned int _phase_s_full_0_2 = 0;
            mbarrier_wait_hint(s_full_addr, _phase_s_full_0_2, 10000000);
            _phase_s_full_0_2 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float slice_max_2 = -CAKE_INF;
            float score_values_2[32];
            unsigned int mask_words_2[4];
            mask_words_2[0] = smem_mask[0];
            mask_words_2[1] = smem_mask[1];
            mask_words_2[2] = smem_mask[2];
            mask_words_2[3] = smem_mask[3];
            if (warp_rows_valid_2 != 0) {
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(score_values_2[0]), "=f"(score_values_2[1]), "=f"(score_values_2[2]), "=f"(score_values_2[3]), "=f"(score_values_2[4]), "=f"(score_values_2[5]), "=f"(score_values_2[6]), "=f"(score_values_2[7]), "=f"(score_values_2[8]), "=f"(score_values_2[9]), "=f"(score_values_2[10]), "=f"(score_values_2[11]), "=f"(score_values_2[12]), "=f"(score_values_2[13]), "=f"(score_values_2[14]), "=f"(score_values_2[15]), "=f"(score_values_2[16]), "=f"(score_values_2[17]), "=f"(score_values_2[18]), "=f"(score_values_2[19]), "=f"(score_values_2[20]), "=f"(score_values_2[21]), "=f"(score_values_2[22]), "=f"(score_values_2[23]), "=f"(score_values_2[24]), "=f"(score_values_2[25]), "=f"(score_values_2[26]), "=f"(score_values_2[27]), "=f"(score_values_2[28]), "=f"(score_values_2[29]), "=f"(score_values_2[30]), "=f"(score_values_2[31])
                    : "r"(taddr + 96 + (unsigned int)(tmem_row_origin_2 << 16)));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int full_mask_2 = mask_words_2[3];
                if (full_mask_2 != 4294967295u) {
                    if ((mask_words_2[3] & 1) == 0) {
                        score_values_2[0] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 1 & 1) == 0) {
                        score_values_2[1] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 2 & 1) == 0) {
                        score_values_2[2] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 3 & 1) == 0) {
                        score_values_2[3] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 4 & 1) == 0) {
                        score_values_2[4] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 5 & 1) == 0) {
                        score_values_2[5] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 6 & 1) == 0) {
                        score_values_2[6] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 7 & 1) == 0) {
                        score_values_2[7] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 8 & 1) == 0) {
                        score_values_2[8] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 9 & 1) == 0) {
                        score_values_2[9] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 10 & 1) == 0) {
                        score_values_2[10] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 11 & 1) == 0) {
                        score_values_2[11] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 12 & 1) == 0) {
                        score_values_2[12] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 13 & 1) == 0) {
                        score_values_2[13] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 14 & 1) == 0) {
                        score_values_2[14] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 15 & 1) == 0) {
                        score_values_2[15] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 16 & 1) == 0) {
                        score_values_2[16] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 17 & 1) == 0) {
                        score_values_2[17] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 18 & 1) == 0) {
                        score_values_2[18] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 19 & 1) == 0) {
                        score_values_2[19] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 20 & 1) == 0) {
                        score_values_2[20] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 21 & 1) == 0) {
                        score_values_2[21] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 22 & 1) == 0) {
                        score_values_2[22] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 23 & 1) == 0) {
                        score_values_2[23] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 24 & 1) == 0) {
                        score_values_2[24] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 25 & 1) == 0) {
                        score_values_2[25] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 26 & 1) == 0) {
                        score_values_2[26] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 27 & 1) == 0) {
                        score_values_2[27] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 28 & 1) == 0) {
                        score_values_2[28] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 29 & 1) == 0) {
                        score_values_2[29] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 30 & 1) == 0) {
                        score_values_2[30] = -CAKE_INF;
                    }
                    if ((mask_words_2[3] >> 31 & 1) == 0) {
                        score_values_2[31] = -CAKE_INF;
                    }
                }
                float2 _reg_reduce_max2_2 = {-CAKE_INF, -CAKE_INF};
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(score_values_2[0], score_values_2[1]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(score_values_2[2], score_values_2[3]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(score_values_2[4], score_values_2[5]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(score_values_2[6], score_values_2[7]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(score_values_2[8], score_values_2[9]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(score_values_2[10], score_values_2[11]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(score_values_2[12], score_values_2[13]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(score_values_2[14], score_values_2[15]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(score_values_2[16], score_values_2[17]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(score_values_2[18], score_values_2[19]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(score_values_2[20], score_values_2[21]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(score_values_2[22], score_values_2[23]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(score_values_2[24], score_values_2[25]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(score_values_2[26], score_values_2[27]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(score_values_2[28], score_values_2[29]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(score_values_2[30], score_values_2[31]));
                float score_values_max_2 = row_max_reduce(_reg_reduce_max2_2);
                slice_max_2 = score_values_max_2;
            }
            smem_pmax[256 + row_2] = slice_max_2;
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            float _max_96 = max_noftz(smem_pmax[row_2], smem_pmax[128 + row_2]);
            float _max_97 = max_noftz(_max_96, smem_pmax[256 + row_2]);
            float tile_max_2 = _max_97;
            float row_max_scaled_2 = tile_max_2 * softmax_scale_log2_2;
            if (has_sink_row_2 != 0) {
                float _max_98 = max_noftz(row_max_scaled_2, sink_log2_2);
                row_max_scaled_2 = _max_98;
            }
            if (row_max_scaled_2 == -CAKE_INF) {
                row_max_scaled_2 = 0.0f;
            }
            float slice_sum_2 = 0.0f;
            float rsum_2 = 0.0f;
            if (warp_rows_valid_2 != 0) {
                float score_bias_2 = -row_max_scaled_2;
                const float2 _fma_b2_3 = {softmax_scale_log2_2, softmax_scale_log2_2};
                const float2 _fma_c2_4 = {score_bias_2, score_bias_2};
                float2 _fma_pair_5 = fma_f32x2(make_float2(score_values_2[0], score_values_2[1]), _fma_b2_3, _fma_c2_4);
                score_values_2[0] = _fma_pair_5.x;
                score_values_2[1] = _fma_pair_5.y;
                float2 _fma_pair_6 = fma_f32x2(make_float2(score_values_2[2], score_values_2[3]), _fma_b2_3, _fma_c2_4);
                score_values_2[2] = _fma_pair_6.x;
                score_values_2[3] = _fma_pair_6.y;
                float2 _fma_pair_7 = fma_f32x2(make_float2(score_values_2[4], score_values_2[5]), _fma_b2_3, _fma_c2_4);
                score_values_2[4] = _fma_pair_7.x;
                score_values_2[5] = _fma_pair_7.y;
                float2 _fma_pair_8 = fma_f32x2(make_float2(score_values_2[6], score_values_2[7]), _fma_b2_3, _fma_c2_4);
                score_values_2[6] = _fma_pair_8.x;
                score_values_2[7] = _fma_pair_8.y;
                float2 _fma_pair_9 = fma_f32x2(make_float2(score_values_2[8], score_values_2[9]), _fma_b2_3, _fma_c2_4);
                score_values_2[8] = _fma_pair_9.x;
                score_values_2[9] = _fma_pair_9.y;
                float2 _fma_pair_10 = fma_f32x2(make_float2(score_values_2[10], score_values_2[11]), _fma_b2_3, _fma_c2_4);
                score_values_2[10] = _fma_pair_10.x;
                score_values_2[11] = _fma_pair_10.y;
                float2 _fma_pair_11 = fma_f32x2(make_float2(score_values_2[12], score_values_2[13]), _fma_b2_3, _fma_c2_4);
                score_values_2[12] = _fma_pair_11.x;
                score_values_2[13] = _fma_pair_11.y;
                float2 _fma_pair_12 = fma_f32x2(make_float2(score_values_2[14], score_values_2[15]), _fma_b2_3, _fma_c2_4);
                score_values_2[14] = _fma_pair_12.x;
                score_values_2[15] = _fma_pair_12.y;
                float2 _fma_pair_13 = fma_f32x2(make_float2(score_values_2[16], score_values_2[17]), _fma_b2_3, _fma_c2_4);
                score_values_2[16] = _fma_pair_13.x;
                score_values_2[17] = _fma_pair_13.y;
                float2 _fma_pair_14 = fma_f32x2(make_float2(score_values_2[18], score_values_2[19]), _fma_b2_3, _fma_c2_4);
                score_values_2[18] = _fma_pair_14.x;
                score_values_2[19] = _fma_pair_14.y;
                float2 _fma_pair_15 = fma_f32x2(make_float2(score_values_2[20], score_values_2[21]), _fma_b2_3, _fma_c2_4);
                score_values_2[20] = _fma_pair_15.x;
                score_values_2[21] = _fma_pair_15.y;
                float2 _fma_pair_16 = fma_f32x2(make_float2(score_values_2[22], score_values_2[23]), _fma_b2_3, _fma_c2_4);
                score_values_2[22] = _fma_pair_16.x;
                score_values_2[23] = _fma_pair_16.y;
                float2 _fma_pair_17 = fma_f32x2(make_float2(score_values_2[24], score_values_2[25]), _fma_b2_3, _fma_c2_4);
                score_values_2[24] = _fma_pair_17.x;
                score_values_2[25] = _fma_pair_17.y;
                float2 _fma_pair_18 = fma_f32x2(make_float2(score_values_2[26], score_values_2[27]), _fma_b2_3, _fma_c2_4);
                score_values_2[26] = _fma_pair_18.x;
                score_values_2[27] = _fma_pair_18.y;
                float2 _fma_pair_19 = fma_f32x2(make_float2(score_values_2[28], score_values_2[29]), _fma_b2_3, _fma_c2_4);
                score_values_2[28] = _fma_pair_19.x;
                score_values_2[29] = _fma_pair_19.y;
                float2 _fma_pair_20 = fma_f32x2(make_float2(score_values_2[30], score_values_2[31]), _fma_b2_3, _fma_c2_4);
                score_values_2[30] = _fma_pair_20.x;
                score_values_2[31] = _fma_pair_20.y;
                #pragma unroll
                for (int _le = 0; _le < 32; _le++) {
                    score_values_2[_le] = approx_exp2(score_values_2[_le]);
                }
                float2 _reg_reduce_sum2_21 = make_float2(0.0f, 0.0f);
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[0], score_values_2[1]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[2], score_values_2[3]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[4], score_values_2[5]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[6], score_values_2[7]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[8], score_values_2[9]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[10], score_values_2[11]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[12], score_values_2[13]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[14], score_values_2[15]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[16], score_values_2[17]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[18], score_values_2[19]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[20], score_values_2[21]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[22], score_values_2[23]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[24], score_values_2[25]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[26], score_values_2[27]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[28], score_values_2[29]));
                _reg_reduce_sum2_21 = add_f32x2(_reg_reduce_sum2_21, make_float2(score_values_2[30], score_values_2[31]));
                float score_values_sum_2 = _reg_reduce_sum2_21.x + _reg_reduce_sum2_21.y;
                slice_sum_2 = score_values_sum_2;
                if (row_valid_2 == 0) {
                    score_values_2[0] = 0.0f;
                    score_values_2[1] = 0.0f;
                    score_values_2[2] = 0.0f;
                    score_values_2[3] = 0.0f;
                    score_values_2[4] = 0.0f;
                    score_values_2[5] = 0.0f;
                    score_values_2[6] = 0.0f;
                    score_values_2[7] = 0.0f;
                    score_values_2[8] = 0.0f;
                    score_values_2[9] = 0.0f;
                    score_values_2[10] = 0.0f;
                    score_values_2[11] = 0.0f;
                    score_values_2[12] = 0.0f;
                    score_values_2[13] = 0.0f;
                    score_values_2[14] = 0.0f;
                    score_values_2[15] = 0.0f;
                    score_values_2[16] = 0.0f;
                    score_values_2[17] = 0.0f;
                    score_values_2[18] = 0.0f;
                    score_values_2[19] = 0.0f;
                    score_values_2[20] = 0.0f;
                    score_values_2[21] = 0.0f;
                    score_values_2[22] = 0.0f;
                    score_values_2[23] = 0.0f;
                    score_values_2[24] = 0.0f;
                    score_values_2[25] = 0.0f;
                    score_values_2[26] = 0.0f;
                    score_values_2[27] = 0.0f;
                    score_values_2[28] = 0.0f;
                    score_values_2[29] = 0.0f;
                    score_values_2[30] = 0.0f;
                    score_values_2[31] = 0.0f;
                }
                unsigned int packed_p_2[8];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(score_values_2[0]), "f"(score_values_2[1]),
                                           "f"(score_values_2[2]), "f"(score_values_2[3]));
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
                        : "=r"(_packed) : "f"(score_values_2[4]), "f"(score_values_2[5]),
                                           "f"(score_values_2[6]), "f"(score_values_2[7]));
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
                        : "=r"(_packed) : "f"(score_values_2[8]), "f"(score_values_2[9]),
                                           "f"(score_values_2[10]), "f"(score_values_2[11]));
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
                        : "=r"(_packed) : "f"(score_values_2[12]), "f"(score_values_2[13]),
                                           "f"(score_values_2[14]), "f"(score_values_2[15]));
                    packed_p_2[3] = _packed;
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
                        : "=r"(_packed) : "f"(score_values_2[16]), "f"(score_values_2[17]),
                                           "f"(score_values_2[18]), "f"(score_values_2[19]));
                    packed_p_2[4] = _packed;
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
                        : "=r"(_packed) : "f"(score_values_2[20]), "f"(score_values_2[21]),
                                           "f"(score_values_2[22]), "f"(score_values_2[23]));
                    packed_p_2[5] = _packed;
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
                        : "=r"(_packed) : "f"(score_values_2[24]), "f"(score_values_2[25]),
                                           "f"(score_values_2[26]), "f"(score_values_2[27]));
                    packed_p_2[6] = _packed;
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
                        : "=r"(_packed) : "f"(score_values_2[28]), "f"(score_values_2[29]),
                                           "f"(score_values_2[30]), "f"(score_values_2[31]));
                    packed_p_2[7] = _packed;
                }
                float _fp8_rt_96;
                uint16_t _e4m3x2_22;
                uint32_t _f16x2_22;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_22) : "f"(0.0f), "f"(score_values_2[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_22) : "h"(_e4m3x2_22));
                uint16_t _fp8_h0_22 = (uint16_t)(_f16x2_22 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_96) : "h"(_fp8_h0_22));
                rsum_2 = rsum_2 + _fp8_rt_96;
                float _fp8_rt_97;
                uint16_t _e4m3x2_23;
                uint32_t _f16x2_23;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_23) : "f"(0.0f), "f"(score_values_2[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_23) : "h"(_e4m3x2_23));
                uint16_t _fp8_h0_23 = (uint16_t)(_f16x2_23 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_97) : "h"(_fp8_h0_23));
                rsum_2 = rsum_2 + _fp8_rt_97;
                float _fp8_rt_98;
                uint16_t _e4m3x2_24;
                uint32_t _f16x2_24;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_24) : "f"(0.0f), "f"(score_values_2[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_24) : "h"(_e4m3x2_24));
                uint16_t _fp8_h0_24 = (uint16_t)(_f16x2_24 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_98) : "h"(_fp8_h0_24));
                rsum_2 = rsum_2 + _fp8_rt_98;
                float _fp8_rt_99;
                uint16_t _e4m3x2_25;
                uint32_t _f16x2_25;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_25) : "f"(0.0f), "f"(score_values_2[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_25) : "h"(_e4m3x2_25));
                uint16_t _fp8_h0_25 = (uint16_t)(_f16x2_25 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_99) : "h"(_fp8_h0_25));
                rsum_2 = rsum_2 + _fp8_rt_99;
                float _fp8_rt_100;
                uint16_t _e4m3x2_26;
                uint32_t _f16x2_26;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_26) : "f"(0.0f), "f"(score_values_2[4]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_26) : "h"(_e4m3x2_26));
                uint16_t _fp8_h0_26 = (uint16_t)(_f16x2_26 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_100) : "h"(_fp8_h0_26));
                rsum_2 = rsum_2 + _fp8_rt_100;
                float _fp8_rt_101;
                uint16_t _e4m3x2_27;
                uint32_t _f16x2_27;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_27) : "f"(0.0f), "f"(score_values_2[5]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_27) : "h"(_e4m3x2_27));
                uint16_t _fp8_h0_27 = (uint16_t)(_f16x2_27 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_101) : "h"(_fp8_h0_27));
                rsum_2 = rsum_2 + _fp8_rt_101;
                float _fp8_rt_102;
                uint16_t _e4m3x2_28;
                uint32_t _f16x2_28;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_28) : "f"(0.0f), "f"(score_values_2[6]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_28) : "h"(_e4m3x2_28));
                uint16_t _fp8_h0_28 = (uint16_t)(_f16x2_28 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_102) : "h"(_fp8_h0_28));
                rsum_2 = rsum_2 + _fp8_rt_102;
                float _fp8_rt_103;
                uint16_t _e4m3x2_29;
                uint32_t _f16x2_29;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_29) : "f"(0.0f), "f"(score_values_2[7]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_29) : "h"(_e4m3x2_29));
                uint16_t _fp8_h0_29 = (uint16_t)(_f16x2_29 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_103) : "h"(_fp8_h0_29));
                rsum_2 = rsum_2 + _fp8_rt_103;
                float _fp8_rt_104;
                uint16_t _e4m3x2_30;
                uint32_t _f16x2_30;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_30) : "f"(0.0f), "f"(score_values_2[8]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_30) : "h"(_e4m3x2_30));
                uint16_t _fp8_h0_30 = (uint16_t)(_f16x2_30 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_104) : "h"(_fp8_h0_30));
                rsum_2 = rsum_2 + _fp8_rt_104;
                float _fp8_rt_105;
                uint16_t _e4m3x2_31;
                uint32_t _f16x2_31;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_31) : "f"(0.0f), "f"(score_values_2[9]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_31) : "h"(_e4m3x2_31));
                uint16_t _fp8_h0_31 = (uint16_t)(_f16x2_31 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_105) : "h"(_fp8_h0_31));
                rsum_2 = rsum_2 + _fp8_rt_105;
                float _fp8_rt_106;
                uint16_t _e4m3x2_32;
                uint32_t _f16x2_32;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_32) : "f"(0.0f), "f"(score_values_2[10]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_32) : "h"(_e4m3x2_32));
                uint16_t _fp8_h0_32 = (uint16_t)(_f16x2_32 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_106) : "h"(_fp8_h0_32));
                rsum_2 = rsum_2 + _fp8_rt_106;
                float _fp8_rt_107;
                uint16_t _e4m3x2_33;
                uint32_t _f16x2_33;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_33) : "f"(0.0f), "f"(score_values_2[11]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_33) : "h"(_e4m3x2_33));
                uint16_t _fp8_h0_33 = (uint16_t)(_f16x2_33 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_107) : "h"(_fp8_h0_33));
                rsum_2 = rsum_2 + _fp8_rt_107;
                float _fp8_rt_108;
                uint16_t _e4m3x2_34;
                uint32_t _f16x2_34;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_34) : "f"(0.0f), "f"(score_values_2[12]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_34) : "h"(_e4m3x2_34));
                uint16_t _fp8_h0_34 = (uint16_t)(_f16x2_34 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_108) : "h"(_fp8_h0_34));
                rsum_2 = rsum_2 + _fp8_rt_108;
                float _fp8_rt_109;
                uint16_t _e4m3x2_35;
                uint32_t _f16x2_35;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_35) : "f"(0.0f), "f"(score_values_2[13]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_35) : "h"(_e4m3x2_35));
                uint16_t _fp8_h0_35 = (uint16_t)(_f16x2_35 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_109) : "h"(_fp8_h0_35));
                rsum_2 = rsum_2 + _fp8_rt_109;
                float _fp8_rt_110;
                uint16_t _e4m3x2_36;
                uint32_t _f16x2_36;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_36) : "f"(0.0f), "f"(score_values_2[14]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_36) : "h"(_e4m3x2_36));
                uint16_t _fp8_h0_36 = (uint16_t)(_f16x2_36 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_110) : "h"(_fp8_h0_36));
                rsum_2 = rsum_2 + _fp8_rt_110;
                float _fp8_rt_111;
                uint16_t _e4m3x2_37;
                uint32_t _f16x2_37;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_37) : "f"(0.0f), "f"(score_values_2[15]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_37) : "h"(_e4m3x2_37));
                uint16_t _fp8_h0_37 = (uint16_t)(_f16x2_37 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_111) : "h"(_fp8_h0_37));
                rsum_2 = rsum_2 + _fp8_rt_111;
                float _fp8_rt_112;
                uint16_t _e4m3x2_38;
                uint32_t _f16x2_38;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_38) : "f"(0.0f), "f"(score_values_2[16]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_38) : "h"(_e4m3x2_38));
                uint16_t _fp8_h0_38 = (uint16_t)(_f16x2_38 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_112) : "h"(_fp8_h0_38));
                rsum_2 = rsum_2 + _fp8_rt_112;
                float _fp8_rt_113;
                uint16_t _e4m3x2_39;
                uint32_t _f16x2_39;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_39) : "f"(0.0f), "f"(score_values_2[17]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_39) : "h"(_e4m3x2_39));
                uint16_t _fp8_h0_39 = (uint16_t)(_f16x2_39 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_113) : "h"(_fp8_h0_39));
                rsum_2 = rsum_2 + _fp8_rt_113;
                float _fp8_rt_114;
                uint16_t _e4m3x2_40;
                uint32_t _f16x2_40;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_40) : "f"(0.0f), "f"(score_values_2[18]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_40) : "h"(_e4m3x2_40));
                uint16_t _fp8_h0_40 = (uint16_t)(_f16x2_40 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_114) : "h"(_fp8_h0_40));
                rsum_2 = rsum_2 + _fp8_rt_114;
                float _fp8_rt_115;
                uint16_t _e4m3x2_41;
                uint32_t _f16x2_41;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_41) : "f"(0.0f), "f"(score_values_2[19]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_41) : "h"(_e4m3x2_41));
                uint16_t _fp8_h0_41 = (uint16_t)(_f16x2_41 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_115) : "h"(_fp8_h0_41));
                rsum_2 = rsum_2 + _fp8_rt_115;
                float _fp8_rt_116;
                uint16_t _e4m3x2_42;
                uint32_t _f16x2_42;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_42) : "f"(0.0f), "f"(score_values_2[20]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_42) : "h"(_e4m3x2_42));
                uint16_t _fp8_h0_42 = (uint16_t)(_f16x2_42 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_116) : "h"(_fp8_h0_42));
                rsum_2 = rsum_2 + _fp8_rt_116;
                float _fp8_rt_117;
                uint16_t _e4m3x2_43;
                uint32_t _f16x2_43;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_43) : "f"(0.0f), "f"(score_values_2[21]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_43) : "h"(_e4m3x2_43));
                uint16_t _fp8_h0_43 = (uint16_t)(_f16x2_43 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_117) : "h"(_fp8_h0_43));
                rsum_2 = rsum_2 + _fp8_rt_117;
                float _fp8_rt_118;
                uint16_t _e4m3x2_44;
                uint32_t _f16x2_44;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_44) : "f"(0.0f), "f"(score_values_2[22]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_44) : "h"(_e4m3x2_44));
                uint16_t _fp8_h0_44 = (uint16_t)(_f16x2_44 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_118) : "h"(_fp8_h0_44));
                rsum_2 = rsum_2 + _fp8_rt_118;
                float _fp8_rt_119;
                uint16_t _e4m3x2_45;
                uint32_t _f16x2_45;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_45) : "f"(0.0f), "f"(score_values_2[23]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_45) : "h"(_e4m3x2_45));
                uint16_t _fp8_h0_45 = (uint16_t)(_f16x2_45 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_119) : "h"(_fp8_h0_45));
                rsum_2 = rsum_2 + _fp8_rt_119;
                float _fp8_rt_120;
                uint16_t _e4m3x2_46;
                uint32_t _f16x2_46;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_46) : "f"(0.0f), "f"(score_values_2[24]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_46) : "h"(_e4m3x2_46));
                uint16_t _fp8_h0_46 = (uint16_t)(_f16x2_46 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_120) : "h"(_fp8_h0_46));
                rsum_2 = rsum_2 + _fp8_rt_120;
                float _fp8_rt_121;
                uint16_t _e4m3x2_47;
                uint32_t _f16x2_47;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_47) : "f"(0.0f), "f"(score_values_2[25]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_47) : "h"(_e4m3x2_47));
                uint16_t _fp8_h0_47 = (uint16_t)(_f16x2_47 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_121) : "h"(_fp8_h0_47));
                rsum_2 = rsum_2 + _fp8_rt_121;
                float _fp8_rt_122;
                uint16_t _e4m3x2_48;
                uint32_t _f16x2_48;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_48) : "f"(0.0f), "f"(score_values_2[26]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_48) : "h"(_e4m3x2_48));
                uint16_t _fp8_h0_48 = (uint16_t)(_f16x2_48 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_122) : "h"(_fp8_h0_48));
                rsum_2 = rsum_2 + _fp8_rt_122;
                float _fp8_rt_123;
                uint16_t _e4m3x2_49;
                uint32_t _f16x2_49;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_49) : "f"(0.0f), "f"(score_values_2[27]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_49) : "h"(_e4m3x2_49));
                uint16_t _fp8_h0_49 = (uint16_t)(_f16x2_49 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_123) : "h"(_fp8_h0_49));
                rsum_2 = rsum_2 + _fp8_rt_123;
                float _fp8_rt_124;
                uint16_t _e4m3x2_50;
                uint32_t _f16x2_50;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_50) : "f"(0.0f), "f"(score_values_2[28]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_50) : "h"(_e4m3x2_50));
                uint16_t _fp8_h0_50 = (uint16_t)(_f16x2_50 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_124) : "h"(_fp8_h0_50));
                rsum_2 = rsum_2 + _fp8_rt_124;
                float _fp8_rt_125;
                uint16_t _e4m3x2_51;
                uint32_t _f16x2_51;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_51) : "f"(0.0f), "f"(score_values_2[29]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_51) : "h"(_e4m3x2_51));
                uint16_t _fp8_h0_51 = (uint16_t)(_f16x2_51 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_125) : "h"(_fp8_h0_51));
                rsum_2 = rsum_2 + _fp8_rt_125;
                float _fp8_rt_126;
                uint16_t _e4m3x2_52;
                uint32_t _f16x2_52;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_52) : "f"(0.0f), "f"(score_values_2[30]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_52) : "h"(_e4m3x2_52));
                uint16_t _fp8_h0_52 = (uint16_t)(_f16x2_52 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_126) : "h"(_fp8_h0_52));
                rsum_2 = rsum_2 + _fp8_rt_126;
                float _fp8_rt_127;
                uint16_t _e4m3x2_53;
                uint32_t _f16x2_53;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_53) : "f"(0.0f), "f"(score_values_2[31]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_53) : "h"(_e4m3x2_53));
                uint16_t _fp8_h0_53 = (uint16_t)(_f16x2_53 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_127) : "h"(_fp8_h0_53));
                rsum_2 = rsum_2 + _fp8_rt_127;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_p_addr + (unsigned int)(row_2 * 128 + (96 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_p_addr + (unsigned int)(row_2 * 128 + (112 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[4])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_p_2[(4) + 3])));
            }
            float sink_term_2 = 0.0f;
            if (has_sink_row_2 != 0) {
                float _exp2_2 = approx_exp2(sink_log2_2 - row_max_scaled_2);
                sink_term_2 = _exp2_2;
            }
            smem_psum[256 + row_2] = slice_sum_2;
            smem_rsum[256 + row_2] = rsum_2;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(p_full_addr);
            asm volatile("barrier.sync 9, 384;" ::: "memory");
            float denom_2 = smem_rsum[row_2] + smem_rsum[128 + row_2] + smem_rsum[256 + row_2] + sink_term_2;
            float _rcp_2 = approx_rcp(denom_2);
            float norm_2 = ((denom_2 > 0.0f) ? _rcp_2 * output_scale_2 : 0.0f);
            float o_values_2[64];
            unsigned int packed_2[32];
            long long out_base_2 = ((long long)(query_idx_2 * num_heads + head_row_2) * (long long)num_splits + (long long)split_idx_2) * 512;
            if (num_heads <= 32) {
            } else if (warp == 8) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            unsigned int _phase_q_ready_0_3 = 0;
            mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0_3, 10000000);
            _phase_q_ready_0_3 ^= 1;
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
            unsigned int _phase_q_rope_full_0 = 0;
            mbarrier_wait_hint(q_rope_full_addr, _phase_q_rope_full_0, 10000000);
            _phase_q_rope_full_0 ^= 1;
            unsigned int _phase_kv_full_0 = 0;
            mbarrier_wait_hint(kv_full_addr, _phase_kv_full_0, 10000000);
            _phase_kv_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (elect_sync()) {
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb0, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4))));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 24)));
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb1, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 24)));
                int _mma_a_lo_0 = ((smem_qrope_addr) >> 4) & 0x3FFF;
                int _mma_b_lo_0 = ((smem_krope_addr) >> 4) & 0x3FFF;
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
                int _mma_b_lo_1 = (((smem_kf4_addr) >> 4) & 0x3FFF) + (0) * 1024;
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
                int _mma_b_lo_2 = (((smem_kf4_addr) >> 4) & 0x3FFF) + (1) * 1024;
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
            unsigned int _phase_p_full_0 = 0;
            mbarrier_wait_hint(p_full_addr, _phase_p_full_0, 10000000);
            _phase_p_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (elect_sync()) {
                int _mma_a_lo_3 = ((smem_p_addr) >> 4) & 0x3FFF;
                int _mma_b_lo_3 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (0) * 1024;
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_tmem_o0), "r"(0));
                tcgen05_commit(o_full_addr);
                int _mma_b_lo_4 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (1) * 1024;
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_4), "r"(tmem_tmem_o1), "r"(0));
                tcgen05_commit(o_full_addr + 8);
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait_hint(tmem_dealloc_addr, _phase_tmem_dealloc_0, 10000000);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 13 && warp <= 15) {
        { // load_warp_main
            const int load_tid = (warp - 13) * 32 + lane;
            int work_idx_3 = blockIdx.x / 2;
            int head_tile_3 = work_idx_3 % num_head_tiles;
            int split_work_3 = work_idx_3 / num_head_tiles;
            int query_idx_3 = split_work_3 / num_splits;
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
        }
    }

    // Cleanup
}

} // extern "C"
