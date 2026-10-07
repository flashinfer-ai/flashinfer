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
#define TMEM_NCOLS 160
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O0_OFFSET 32
#define TMEM_TMEM_O1_OFFSET 64
#define TMEM_TMEM_O2_OFFSET 96
#define TMEM_TMEM_O3_OFFSET 128
#define TMEM_TMEM_SFA0_OFFSET 64
#define TMEM_TMEM_SFA1_OFFSET 80
#define TMEM_TMEM_SFB0_OFFSET 96
#define TMEM_TMEM_SFB1_OFFSET 112
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
#define SMEM_SMEM_QROPE_STAGE_BYTES 4096
#define SMEM_SMEM_QROPE_STRIDE 4096
#define SMEM_SMEM_QSTAGE_OFF 123904
#define SMEM_SMEM_QSTAGE_STAGE_BYTES 4096
#define SMEM_SMEM_QSTAGE_STRIDE 4096
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
#define SMEM_SMEM_SFS_OFF 152576
#define SMEM_SMEM_SFS_STAGE_BYTES 4096
#define SMEM_SMEM_SFS_STRIDE 4096
#define SMEM_SMEM_P_OFF 123904
#define SMEM_SMEM_P_STAGE_BYTES 16384
#define SMEM_SMEM_P_STRIDE 16384
#define SMEM_SMEM_RCPTAB_OFF 156672
#define SMEM_SMEM_RCPTAB_STAGE_BYTES 64
#define SMEM_SMEM_RCPTAB_STRIDE 64
#define SMEM_SMEM_MASK_OFF 158720
#define SMEM_SMEM_MASK_STAGE_BYTES 16
#define SMEM_SMEM_MASK_STRIDE 16
#define SMEM_SMEM_TOK_OFF 158752
#define SMEM_SMEM_TOK_STAGE_BYTES 512
#define SMEM_SMEM_TOK_STRIDE 512
#define SMEM_SMEM_ROWOFF_OFF 162080
#define SMEM_SMEM_ROWOFF_STAGE_BYTES 2048
#define SMEM_SMEM_ROWOFF_STRIDE 2048
#define SMEM_SMEM_SFS32_OFF 152576
#define SMEM_SMEM_SFS32_STAGE_BYTES 4096
#define SMEM_SMEM_SFS32_STRIDE 4096
#define SMEM_SMEM_PMAX_OFF 159776
#define SMEM_SMEM_PMAX_STAGE_BYTES 768
#define SMEM_SMEM_PMAX_STRIDE 768
#define SMEM_SMEM_PSUM_OFF 160544
#define SMEM_SMEM_PSUM_STAGE_BYTES 768
#define SMEM_SMEM_PSUM_STRIDE 768
#define SMEM_SMEM_RSUM_OFF 161312
#define SMEM_SMEM_RSUM_STAGE_BYTES 768
#define SMEM_SMEM_RSUM_STRIDE 768
#define SMEM_SMEM_QF4B_OFF 1024
#define SMEM_SMEM_QF4B_STAGE_BYTES 4096
#define SMEM_SMEM_QF4B_STRIDE 16384
#define SMEM_SMEM_QROPE_B_OFF 37888
#define SMEM_SMEM_QROPE_B_STAGE_BYTES 4096
#define SMEM_SMEM_QROPE_B_STRIDE 4096
#define SMEM_SMEM_PB_OFF 123904
#define SMEM_SMEM_PB_STAGE_BYTES 4096
#define SMEM_SMEM_PB_STRIDE 4096
#define SMEM_TOTAL 164224
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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



__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsv4_nvfp4_74d41ccae5030cee3db7(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_out, __nv_bfloat16* __restrict__ q_rows, uint8_t* __restrict__ main_cache, uint8_t* __restrict__ extra_cache, int* __restrict__ main_indices, int* __restrict__ extra_indices, int* __restrict__ main_lengths, int* __restrict__ extra_lengths, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ lse_out, int num_heads, int num_head_tiles, int num_splits, int num_main_tiles, int main_width, int extra_width, int main_index_stride, int extra_index_stride, int has_main_lengths, int has_extra_lengths, int main_page_shift, int extra_page_shift, long long main_page_stride, long long extra_page_stride, int has_sinks, float lse_partial_scale, float lse_scale)
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
    #define tmem_dealloc_addr (mbar_base + 72)

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
    __nv_bfloat16* smem_qstage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 123904);
    const int smem_qstage_addr = smem + 123904;
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
    uint8_t* smem_sfs = reinterpret_cast<uint8_t*>(smem_raw + 152576);
    const int smem_sfs_addr = smem + 152576;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 123904);
    const int smem_p_addr = smem + 123904;
    unsigned int* smem_rcptab = reinterpret_cast<unsigned int*>(smem_raw + 156672);
    const int smem_rcptab_addr = smem + 156672;
    unsigned int* smem_mask = reinterpret_cast<unsigned int*>(smem_raw + 158720);
    const int smem_mask_addr = smem + 158720;
    int* smem_tok = reinterpret_cast<int*>(smem_raw + 158752);
    const int smem_tok_addr = smem + 158752;
    unsigned int* smem_rowoff = reinterpret_cast<unsigned int*>(smem_raw + 162080);
    const int smem_rowoff_addr = smem + 162080;
    unsigned int* smem_sfs32 = reinterpret_cast<unsigned int*>(smem_raw + 152576);
    const int smem_sfs32_addr = smem + 152576;
    float* smem_pmax = reinterpret_cast<float*>(smem_raw + 159776);
    const int smem_pmax_addr = smem + 159776;
    float* smem_psum = reinterpret_cast<float*>(smem_raw + 160544);
    const int smem_psum_addr = smem + 160544;
    float* smem_rsum = reinterpret_cast<float*>(smem_raw + 161312);
    const int smem_rsum_addr = smem + 161312;
    uint8_t* smem_qf4b = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_qf4b_addr = smem + 1024;
    __nv_bfloat16* smem_qrope_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 37888);
    const int smem_qrope_b_addr = smem + 37888;
    uint8_t* smem_pb = reinterpret_cast<uint8_t*>(smem_raw + 123904);
    const int smem_pb_addr = smem + 123904;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_out))) : "memory");

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

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
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // tmem_dealloc: 1 barriers, init_count=384
            mbarrier_init(smem + 72, 384);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 160 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 0) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o0 = taddr + 32;
    const int tmem_tmem_o1 = taddr + 64;
    const int tmem_tmem_o2 = taddr + 96;
    const int tmem_tmem_o3 = taddr + 128;
    const int tmem_tmem_sfa0 = taddr + 64;
    const int tmem_tmem_sfa1 = taddr + 80;
    const int tmem_tmem_sfb0 = taddr + 96;
    const int tmem_tmem_sfb1 = taddr + 112;

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
            int o_chunk = blockIdx.x % 4;
            int work_idx = blockIdx.x / 4;
            int head_tile = work_idx % num_head_tiles;
            int split_work = work_idx / num_head_tiles;
            int split_idx = split_work % num_splits;
            int query_idx = split_work / num_splits;
            int head_base = head_tile * 128;
            const int row = local_warp * 32 + lane;
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
            int strip = smem_sfs_addr + (unsigned int)(row * 32);
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
            const int g_kind = ((g_chunk < 14) ? 0 : ((g_chunk < 22) ? 1 : 2));
            long long g_src_off = (long long)(((g_kind < 2) ? 16 * g_chunk : 16 * (g_chunk - 14 - 8)));
            int g_dst_k = smem_kf4_addr + (unsigned int)(g_chunk / 8 * 16384) + (unsigned int)(g_row0 * 128 + (g_chunk % 8 * 16 ^ g_row0 % 8 * 16));
            int g_dst_r = smem_krope_addr + (unsigned int)(g_row0 * 128 + ((g_chunk - 14) * 16 ^ g_row0 % 8 * 16));
            int g_dst_s = smem_sfs_addr + (unsigned int)(g_row0 * 32) + (unsigned int)(16 * (g_chunk - 14 - 8));
            int g_dst0 = ((g_kind == 0) ? g_dst_k : ((g_kind == 1) ? g_dst_r : g_dst_s));
            const int g_row_bytes = ((g_kind < 2) ? 128 : 32);
            unsigned int w4[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)(g_row0 * 16)));
            long long od = (long long)w4[1] << 32 | (long long)w4[0];
            long long osf = (long long)w4[3] << 32 | (long long)w4[2];
            long long goff = ((g_kind < 2) ? od : osf);
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
            long long goff_3 = ((g_kind < 2) ? od_1 : osf_2);
            if (goff_3 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 16 * g_row_bytes), "l"(cache + (goff_3 + g_src_off)));
            }
            unsigned int w4_4[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 32) * 16)));
            long long od_5 = (long long)w4_4[1] << 32 | (long long)w4_4[0];
            long long osf_6 = (long long)w4_4[3] << 32 | (long long)w4_4[2];
            long long goff_7 = ((g_kind < 2) ? od_5 : osf_6);
            if (goff_7 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 32 * g_row_bytes), "l"(cache + (goff_7 + g_src_off)));
            }
            unsigned int w4_8[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 48) * 16)));
            long long od_9 = (long long)w4_8[1] << 32 | (long long)w4_8[0];
            long long osf_10 = (long long)w4_8[3] << 32 | (long long)w4_8[2];
            long long goff_11 = ((g_kind < 2) ? od_9 : osf_10);
            if (goff_11 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 48 * g_row_bytes), "l"(cache + (goff_11 + g_src_off)));
            }
            unsigned int w4_12[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 64) * 16)));
            long long od_13 = (long long)w4_12[1] << 32 | (long long)w4_12[0];
            long long osf_14 = (long long)w4_12[3] << 32 | (long long)w4_12[2];
            long long goff_15 = ((g_kind < 2) ? od_13 : osf_14);
            if (goff_15 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 64 * g_row_bytes), "l"(cache + (goff_15 + g_src_off)));
            }
            unsigned int w4_16[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 80) * 16)));
            long long od_17 = (long long)w4_16[1] << 32 | (long long)w4_16[0];
            long long osf_18 = (long long)w4_16[3] << 32 | (long long)w4_16[2];
            long long goff_19 = ((g_kind < 2) ? od_17 : osf_18);
            if (goff_19 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 80 * g_row_bytes), "l"(cache + (goff_19 + g_src_off)));
            }
            unsigned int w4_20[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 96) * 16)));
            long long od_21 = (long long)w4_20[1] << 32 | (long long)w4_20[0];
            long long osf_22 = (long long)w4_20[3] << 32 | (long long)w4_20[2];
            long long goff_23 = ((g_kind < 2) ? od_21 : osf_22);
            if (goff_23 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 96 * g_row_bytes), "l"(cache + (goff_23 + g_src_off)));
            }
            unsigned int w4_24[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0 + 112) * 16)));
            long long od_25 = (long long)w4_24[1] << 32 | (long long)w4_24[0];
            long long osf_26 = (long long)w4_24[3] << 32 | (long long)w4_24[2];
            long long goff_27 = ((g_kind < 2) ? od_25 : osf_26);
            if (goff_27 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0 + 112 * g_row_bytes), "l"(cache + (goff_27 + g_src_off)));
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
                    if (head_base + q_block * 32 < num_heads && q_row < 32) {
                        int q_row_addr = smem_qstage_addr + (unsigned int)(kset * 4096) + (unsigned int)(q_row * 128);
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
            asm volatile("cp.async.wait_group 0;");
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            if (valid != 0) {
                {
                    unsigned int sfw[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw[(0) + 3]))
                        : "r"(strip));
                    smem_ksf32[row % 32 / 8 * 512 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[0];
                    smem_ksf32[row % 32 / 8 * 512 + 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[1];
                    smem_ksf32[row % 32 / 8 * 512 + 256 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[2];
                    smem_ksf32[row % 32 / 8 * 512 + 384 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = sfw[3];
                }
                int vblock = 8 * o_chunk;
                unsigned int v8[4];
                {
                    int vchunk = vblock >> 1;
                    int vhalf = vblock & 1;
                    unsigned int kraw[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk / 8 * 16384) + (unsigned int)(row * 128 + (vchunk % 8 * 16 ^ row % 8 * 16)) + (unsigned int)(8 * vhalf)));
                    unsigned int sfw32 = smem_sfs32[row * 32 + vblock >> 2];
                    unsigned int scale = sfw32 >> (unsigned int)(8 * (vblock & 3)) & 255;
                    {
                        v8[0] = cake_dsv4_qmul4_portable<5>(kraw[0], scale);
                    }
                    {
                        v8[1] = cake_dsv4_qmul4_portable<6>(kraw[0], scale);
                    }
                    {
                        v8[2] = cake_dsv4_qmul4_portable<5>(kraw[1], scale);
                    }
                    {
                        v8[3] = cake_dsv4_qmul4_portable<6>(kraw[1], scale);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row * 128 + (0 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8[0])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8[(0) + 3])));
                int vblock_0 = 8 * o_chunk + 1;
                unsigned int v8_1[4];
                {
                    int vchunk_1 = vblock_0 >> 1;
                    int vhalf_1 = vblock_0 & 1;
                    unsigned int kraw_1[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_1[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_1 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_1 % 8 * 16 ^ row % 8 * 16)) + (unsigned int)(8 * vhalf_1)));
                    unsigned int sfw32_1 = smem_sfs32[row * 32 + vblock_0 >> 2];
                    unsigned int scale_1 = sfw32_1 >> (unsigned int)(8 * (vblock_0 & 3)) & 255;
                    {
                        v8_1[0] = cake_dsv4_qmul4_portable<5>(kraw_1[0], scale_1);
                    }
                    {
                        v8_1[1] = cake_dsv4_qmul4_portable<6>(kraw_1[0], scale_1);
                    }
                    {
                        v8_1[2] = cake_dsv4_qmul4_portable<5>(kraw_1[1], scale_1);
                    }
                    {
                        v8_1[3] = cake_dsv4_qmul4_portable<6>(kraw_1[1], scale_1);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row * 128 + (16 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1[(0) + 3])));
                int vblock_2 = 8 * o_chunk + 2;
                unsigned int v8_3[4];
                {
                    int vchunk_2 = vblock_2 >> 1;
                    int vhalf_2 = vblock_2 & 1;
                    unsigned int kraw_2[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_2[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_2 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_2 % 8 * 16 ^ row % 8 * 16)) + (unsigned int)(8 * vhalf_2)));
                    unsigned int sfw32_2 = smem_sfs32[row * 32 + vblock_2 >> 2];
                    unsigned int scale_2 = sfw32_2 >> (unsigned int)(8 * (vblock_2 & 3)) & 255;
                    {
                        v8_3[0] = cake_dsv4_qmul4_portable<5>(kraw_2[0], scale_2);
                    }
                    {
                        v8_3[1] = cake_dsv4_qmul4_portable<6>(kraw_2[0], scale_2);
                    }
                    {
                        v8_3[2] = cake_dsv4_qmul4_portable<5>(kraw_2[1], scale_2);
                    }
                    {
                        v8_3[3] = cake_dsv4_qmul4_portable<6>(kraw_2[1], scale_2);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row * 128 + (32 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_3[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3[(0) + 3])));
            } else {
                {
                    smem_ksf32[row % 32 / 8 * 512 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[row % 32 / 8 * 512 + 128 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[row % 32 / 8 * 512 + 256 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[row % 32 / 8 * 512 + 384 + row % 8 * 16 + row / 32 % 4 * 4 >> 2] = 0;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (0 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (16 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (32 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_o_full_0 = 0;
            {
                mbarrier_wait_hint(s_full_addr, _phase_s_full_0, 10000000);
                _phase_s_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float score_values[16];
                tmem_ld_x16(&score_values[0], taddr + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (valid == 0) {
                    score_values[0] = -CAKE_INF;
                    score_values[1] = -CAKE_INF;
                    score_values[2] = -CAKE_INF;
                    score_values[3] = -CAKE_INF;
                    score_values[4] = -CAKE_INF;
                    score_values[5] = -CAKE_INF;
                    score_values[6] = -CAKE_INF;
                    score_values[7] = -CAKE_INF;
                    score_values[8] = -CAKE_INF;
                    score_values[9] = -CAKE_INF;
                    score_values[10] = -CAKE_INF;
                    score_values[11] = -CAKE_INF;
                    score_values[12] = -CAKE_INF;
                    score_values[13] = -CAKE_INF;
                    score_values[14] = -CAKE_INF;
                    score_values[15] = -CAKE_INF;
                }
                float tr_vals[16];
                tr_vals[0] = score_values[0];
                tr_vals[1] = score_values[1];
                tr_vals[2] = score_values[2];
                tr_vals[3] = score_values[3];
                tr_vals[4] = score_values[4];
                tr_vals[5] = score_values[5];
                tr_vals[6] = score_values[6];
                tr_vals[7] = score_values[7];
                tr_vals[8] = score_values[8];
                tr_vals[9] = score_values[9];
                tr_vals[10] = score_values[10];
                tr_vals[11] = score_values[11];
                tr_vals[12] = score_values[12];
                tr_vals[13] = score_values[13];
                tr_vals[14] = score_values[14];
                tr_vals[15] = score_values[15];
                int hi_bit = lane & 16;
                float send = ((hi_bit != 0) ? tr_vals[0] : tr_vals[8]);
                float keep = ((hi_bit != 0) ? tr_vals[8] : tr_vals[0]);
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, send, lane ^ 16);
                float recv = _shfl_0;
                float _max_30 = max_noftz(keep, recv);
                tr_vals[0] = _max_30;
                float send_0 = ((hi_bit != 0) ? tr_vals[1] : tr_vals[9]);
                float keep_1 = ((hi_bit != 0) ? tr_vals[9] : tr_vals[1]);
                float _shfl_1 = __shfl_sync(0xFFFFFFFF, send_0, lane ^ 16);
                float recv_2 = _shfl_1;
                float _max_31 = max_noftz(keep_1, recv_2);
                tr_vals[1] = _max_31;
                float send_3 = ((hi_bit != 0) ? tr_vals[2] : tr_vals[10]);
                float keep_4 = ((hi_bit != 0) ? tr_vals[10] : tr_vals[2]);
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, send_3, lane ^ 16);
                float recv_5 = _shfl_2;
                float _max_32 = max_noftz(keep_4, recv_5);
                tr_vals[2] = _max_32;
                float send_6 = ((hi_bit != 0) ? tr_vals[3] : tr_vals[11]);
                float keep_7 = ((hi_bit != 0) ? tr_vals[11] : tr_vals[3]);
                float _shfl_3 = __shfl_sync(0xFFFFFFFF, send_6, lane ^ 16);
                float recv_8 = _shfl_3;
                float _max_33 = max_noftz(keep_7, recv_8);
                tr_vals[3] = _max_33;
                float send_9 = ((hi_bit != 0) ? tr_vals[4] : tr_vals[12]);
                float keep_10 = ((hi_bit != 0) ? tr_vals[12] : tr_vals[4]);
                float _shfl_4 = __shfl_sync(0xFFFFFFFF, send_9, lane ^ 16);
                float recv_11 = _shfl_4;
                float _max_34 = max_noftz(keep_10, recv_11);
                tr_vals[4] = _max_34;
                float send_12 = ((hi_bit != 0) ? tr_vals[5] : tr_vals[13]);
                float keep_13 = ((hi_bit != 0) ? tr_vals[13] : tr_vals[5]);
                float _shfl_5 = __shfl_sync(0xFFFFFFFF, send_12, lane ^ 16);
                float recv_14 = _shfl_5;
                float _max_35 = max_noftz(keep_13, recv_14);
                tr_vals[5] = _max_35;
                float send_15 = ((hi_bit != 0) ? tr_vals[6] : tr_vals[14]);
                float keep_16 = ((hi_bit != 0) ? tr_vals[14] : tr_vals[6]);
                float _shfl_6 = __shfl_sync(0xFFFFFFFF, send_15, lane ^ 16);
                float recv_17 = _shfl_6;
                float _max_36 = max_noftz(keep_16, recv_17);
                tr_vals[6] = _max_36;
                float send_18 = ((hi_bit != 0) ? tr_vals[7] : tr_vals[15]);
                float keep_19 = ((hi_bit != 0) ? tr_vals[15] : tr_vals[7]);
                float _shfl_7 = __shfl_sync(0xFFFFFFFF, send_18, lane ^ 16);
                float recv_20 = _shfl_7;
                float _max_37 = max_noftz(keep_19, recv_20);
                tr_vals[7] = _max_37;
                int hi_bit_21 = lane & 8;
                float send_22 = ((hi_bit_21 != 0) ? tr_vals[0] : tr_vals[4]);
                float keep_23 = ((hi_bit_21 != 0) ? tr_vals[4] : tr_vals[0]);
                float _shfl_8 = __shfl_sync(0xFFFFFFFF, send_22, lane ^ 8);
                float recv_24 = _shfl_8;
                float _max_38 = max_noftz(keep_23, recv_24);
                tr_vals[0] = _max_38;
                float send_25 = ((hi_bit_21 != 0) ? tr_vals[1] : tr_vals[5]);
                float keep_26 = ((hi_bit_21 != 0) ? tr_vals[5] : tr_vals[1]);
                float _shfl_9 = __shfl_sync(0xFFFFFFFF, send_25, lane ^ 8);
                float recv_27 = _shfl_9;
                float _max_39 = max_noftz(keep_26, recv_27);
                tr_vals[1] = _max_39;
                float send_28 = ((hi_bit_21 != 0) ? tr_vals[2] : tr_vals[6]);
                float keep_29 = ((hi_bit_21 != 0) ? tr_vals[6] : tr_vals[2]);
                float _shfl_10 = __shfl_sync(0xFFFFFFFF, send_28, lane ^ 8);
                float recv_30 = _shfl_10;
                float _max_40 = max_noftz(keep_29, recv_30);
                tr_vals[2] = _max_40;
                float send_31 = ((hi_bit_21 != 0) ? tr_vals[3] : tr_vals[7]);
                float keep_32 = ((hi_bit_21 != 0) ? tr_vals[7] : tr_vals[3]);
                float _shfl_11 = __shfl_sync(0xFFFFFFFF, send_31, lane ^ 8);
                float recv_33 = _shfl_11;
                float _max_41 = max_noftz(keep_32, recv_33);
                tr_vals[3] = _max_41;
                int hi_bit_34 = lane & 4;
                float send_35 = ((hi_bit_34 != 0) ? tr_vals[0] : tr_vals[2]);
                float keep_36 = ((hi_bit_34 != 0) ? tr_vals[2] : tr_vals[0]);
                float _shfl_12 = __shfl_sync(0xFFFFFFFF, send_35, lane ^ 4);
                float recv_37 = _shfl_12;
                float _max_42 = max_noftz(keep_36, recv_37);
                tr_vals[0] = _max_42;
                float send_38 = ((hi_bit_34 != 0) ? tr_vals[1] : tr_vals[3]);
                float keep_39 = ((hi_bit_34 != 0) ? tr_vals[3] : tr_vals[1]);
                float _shfl_13 = __shfl_sync(0xFFFFFFFF, send_38, lane ^ 4);
                float recv_40 = _shfl_13;
                float _max_43 = max_noftz(keep_39, recv_40);
                tr_vals[1] = _max_43;
                int hi_bit_41 = lane & 2;
                float send_42 = ((hi_bit_41 != 0) ? tr_vals[0] : tr_vals[1]);
                float keep_43 = ((hi_bit_41 != 0) ? tr_vals[1] : tr_vals[0]);
                float _shfl_14 = __shfl_sync(0xFFFFFFFF, send_42, lane ^ 2);
                float recv_44 = _shfl_14;
                float _max_44 = max_noftz(keep_43, recv_44);
                tr_vals[0] = _max_44;
                float _shfl_15 = __shfl_sync(0xFFFFFFFF, tr_vals[0], lane ^ 1);
                float other = _shfl_15;
                float _max_45 = max_noftz(tr_vals[0], other);
                tr_vals[0] = _max_45;
                if ((lane & 1) == 0) {
                    smem_pmax[local_warp * 16 + (lane >> 1)] = tr_vals[0];
                }
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                float m_lane = -CAKE_INF;
                float sink_lane = -CAKE_INF;
                float m_scaled = 0.0f;
                if (lane < 16) {
                    float _max_46 = max_noftz(smem_pmax[lane], smem_pmax[16 + lane]);
                    float _max_47 = max_noftz(smem_pmax[32 + lane], smem_pmax[48 + lane]);
                    float _max_48 = max_noftz(_max_46, _max_47);
                    m_lane = _max_48;
                    if (has_sinks != 0 && split_idx == 0 && head_base + lane < num_heads) {
                        sink_lane = sinks[head_base + lane] * 1.4426950408889634f;
                    }
                    float _max_49 = max_noftz(m_lane * softmax_scale_log2, sink_lane);
                    m_scaled = _max_49;
                    if (m_scaled == -CAKE_INF) {
                        m_scaled = 0.0f;
                    }
                }
                float col_max[16];
                float _shfl_16 = __shfl_sync(0xFFFFFFFF, m_scaled, 0);
                col_max[0] = _shfl_16;
                float _shfl_17 = __shfl_sync(0xFFFFFFFF, m_scaled, 1);
                col_max[1] = _shfl_17;
                float _shfl_18 = __shfl_sync(0xFFFFFFFF, m_scaled, 2);
                col_max[2] = _shfl_18;
                float _shfl_19 = __shfl_sync(0xFFFFFFFF, m_scaled, 3);
                col_max[3] = _shfl_19;
                float _shfl_20 = __shfl_sync(0xFFFFFFFF, m_scaled, 4);
                col_max[4] = _shfl_20;
                float _shfl_21 = __shfl_sync(0xFFFFFFFF, m_scaled, 5);
                col_max[5] = _shfl_21;
                float _shfl_22 = __shfl_sync(0xFFFFFFFF, m_scaled, 6);
                col_max[6] = _shfl_22;
                float _shfl_23 = __shfl_sync(0xFFFFFFFF, m_scaled, 7);
                col_max[7] = _shfl_23;
                float _shfl_24 = __shfl_sync(0xFFFFFFFF, m_scaled, 8);
                col_max[8] = _shfl_24;
                float _shfl_25 = __shfl_sync(0xFFFFFFFF, m_scaled, 9);
                col_max[9] = _shfl_25;
                float _shfl_26 = __shfl_sync(0xFFFFFFFF, m_scaled, 10);
                col_max[10] = _shfl_26;
                float _shfl_27 = __shfl_sync(0xFFFFFFFF, m_scaled, 11);
                col_max[11] = _shfl_27;
                float _shfl_28 = __shfl_sync(0xFFFFFFFF, m_scaled, 12);
                col_max[12] = _shfl_28;
                float _shfl_29 = __shfl_sync(0xFFFFFFFF, m_scaled, 13);
                col_max[13] = _shfl_29;
                float _shfl_30 = __shfl_sync(0xFFFFFFFF, m_scaled, 14);
                col_max[14] = _shfl_30;
                float _shfl_31 = __shfl_sync(0xFFFFFFFF, m_scaled, 15);
                col_max[15] = _shfl_31;
                float _exp2_0 = approx_exp2(score_values[0] * softmax_scale_log2 - col_max[0]);
                score_values[0] = _exp2_0;
                float _exp2_1 = approx_exp2(score_values[1] * softmax_scale_log2 - col_max[1]);
                score_values[1] = _exp2_1;
                float _exp2_2 = approx_exp2(score_values[2] * softmax_scale_log2 - col_max[2]);
                score_values[2] = _exp2_2;
                float _exp2_3 = approx_exp2(score_values[3] * softmax_scale_log2 - col_max[3]);
                score_values[3] = _exp2_3;
                float _exp2_4 = approx_exp2(score_values[4] * softmax_scale_log2 - col_max[4]);
                score_values[4] = _exp2_4;
                float _exp2_5 = approx_exp2(score_values[5] * softmax_scale_log2 - col_max[5]);
                score_values[5] = _exp2_5;
                float _exp2_6 = approx_exp2(score_values[6] * softmax_scale_log2 - col_max[6]);
                score_values[6] = _exp2_6;
                float _exp2_7 = approx_exp2(score_values[7] * softmax_scale_log2 - col_max[7]);
                score_values[7] = _exp2_7;
                float _exp2_8 = approx_exp2(score_values[8] * softmax_scale_log2 - col_max[8]);
                score_values[8] = _exp2_8;
                float _exp2_9 = approx_exp2(score_values[9] * softmax_scale_log2 - col_max[9]);
                score_values[9] = _exp2_9;
                float _exp2_10 = approx_exp2(score_values[10] * softmax_scale_log2 - col_max[10]);
                score_values[10] = _exp2_10;
                float _exp2_11 = approx_exp2(score_values[11] * softmax_scale_log2 - col_max[11]);
                score_values[11] = _exp2_11;
                float _exp2_12 = approx_exp2(score_values[12] * softmax_scale_log2 - col_max[12]);
                score_values[12] = _exp2_12;
                float _exp2_13 = approx_exp2(score_values[13] * softmax_scale_log2 - col_max[13]);
                score_values[13] = _exp2_13;
                float _exp2_14 = approx_exp2(score_values[14] * softmax_scale_log2 - col_max[14]);
                score_values[14] = _exp2_14;
                float _exp2_15 = approx_exp2(score_values[15] * softmax_scale_log2 - col_max[15]);
                score_values[15] = _exp2_15;
                {
                    uint16_t _fp8_pair_14;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_14) : "f"(0.0f), "f"(score_values[0]));
                    uint32_t _byte_14 = (uint32_t)(_fp8_pair_14 & 0xFF);
                    uint32_t _addr_14 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(row ^ (row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_14), "r"(_byte_14) : "memory");
                }
                float _fp8_rt_0;
                uint16_t _e4m3x2_15;
                uint32_t _f16x2_15;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_15) : "f"(0.0f), "f"(score_values[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_15) : "h"(_e4m3x2_15));
                uint16_t _fp8_h0_15 = (uint16_t)(_f16x2_15 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_15));
                tr_vals[0] = _fp8_rt_0;
                {
                    uint16_t _fp8_pair_16;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_16) : "f"(0.0f), "f"(score_values[1]));
                    uint32_t _byte_16 = (uint32_t)(_fp8_pair_16 & 0xFF);
                    uint32_t _addr_16 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(128 + row ^ (128 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_16), "r"(_byte_16) : "memory");
                }
                float _fp8_rt_1;
                uint16_t _e4m3x2_17;
                uint32_t _f16x2_17;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_17) : "f"(0.0f), "f"(score_values[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_17) : "h"(_e4m3x2_17));
                uint16_t _fp8_h0_17 = (uint16_t)(_f16x2_17 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_17));
                tr_vals[1] = _fp8_rt_1;
                {
                    uint16_t _fp8_pair_18;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_18) : "f"(0.0f), "f"(score_values[2]));
                    uint32_t _byte_18 = (uint32_t)(_fp8_pair_18 & 0xFF);
                    uint32_t _addr_18 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(256 + row ^ (256 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_18), "r"(_byte_18) : "memory");
                }
                float _fp8_rt_2;
                uint16_t _e4m3x2_19;
                uint32_t _f16x2_19;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_19) : "f"(0.0f), "f"(score_values[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_19) : "h"(_e4m3x2_19));
                uint16_t _fp8_h0_19 = (uint16_t)(_f16x2_19 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_2) : "h"(_fp8_h0_19));
                tr_vals[2] = _fp8_rt_2;
                {
                    uint16_t _fp8_pair_20;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_20) : "f"(0.0f), "f"(score_values[3]));
                    uint32_t _byte_20 = (uint32_t)(_fp8_pair_20 & 0xFF);
                    uint32_t _addr_20 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(384 + row ^ (384 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_20), "r"(_byte_20) : "memory");
                }
                float _fp8_rt_3;
                uint16_t _e4m3x2_21;
                uint32_t _f16x2_21;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_21) : "f"(0.0f), "f"(score_values[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_21) : "h"(_e4m3x2_21));
                uint16_t _fp8_h0_21 = (uint16_t)(_f16x2_21 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_3) : "h"(_fp8_h0_21));
                tr_vals[3] = _fp8_rt_3;
                {
                    uint16_t _fp8_pair_22;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_22) : "f"(0.0f), "f"(score_values[4]));
                    uint32_t _byte_22 = (uint32_t)(_fp8_pair_22 & 0xFF);
                    uint32_t _addr_22 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(512 + row ^ (512 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_22), "r"(_byte_22) : "memory");
                }
                float _fp8_rt_4;
                uint16_t _e4m3x2_23;
                uint32_t _f16x2_23;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_23) : "f"(0.0f), "f"(score_values[4]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_23) : "h"(_e4m3x2_23));
                uint16_t _fp8_h0_23 = (uint16_t)(_f16x2_23 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_4) : "h"(_fp8_h0_23));
                tr_vals[4] = _fp8_rt_4;
                {
                    uint16_t _fp8_pair_24;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_24) : "f"(0.0f), "f"(score_values[5]));
                    uint32_t _byte_24 = (uint32_t)(_fp8_pair_24 & 0xFF);
                    uint32_t _addr_24 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(640 + row ^ (640 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_24), "r"(_byte_24) : "memory");
                }
                float _fp8_rt_5;
                uint16_t _e4m3x2_25;
                uint32_t _f16x2_25;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_25) : "f"(0.0f), "f"(score_values[5]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_25) : "h"(_e4m3x2_25));
                uint16_t _fp8_h0_25 = (uint16_t)(_f16x2_25 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_5) : "h"(_fp8_h0_25));
                tr_vals[5] = _fp8_rt_5;
                {
                    uint16_t _fp8_pair_26;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_26) : "f"(0.0f), "f"(score_values[6]));
                    uint32_t _byte_26 = (uint32_t)(_fp8_pair_26 & 0xFF);
                    uint32_t _addr_26 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(768 + row ^ (768 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_26), "r"(_byte_26) : "memory");
                }
                float _fp8_rt_6;
                uint16_t _e4m3x2_27;
                uint32_t _f16x2_27;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_27) : "f"(0.0f), "f"(score_values[6]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_27) : "h"(_e4m3x2_27));
                uint16_t _fp8_h0_27 = (uint16_t)(_f16x2_27 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_6) : "h"(_fp8_h0_27));
                tr_vals[6] = _fp8_rt_6;
                {
                    uint16_t _fp8_pair_28;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_28) : "f"(0.0f), "f"(score_values[7]));
                    uint32_t _byte_28 = (uint32_t)(_fp8_pair_28 & 0xFF);
                    uint32_t _addr_28 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(896 + row ^ (896 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_28), "r"(_byte_28) : "memory");
                }
                float _fp8_rt_7;
                uint16_t _e4m3x2_29;
                uint32_t _f16x2_29;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_29) : "f"(0.0f), "f"(score_values[7]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_29) : "h"(_e4m3x2_29));
                uint16_t _fp8_h0_29 = (uint16_t)(_f16x2_29 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_7) : "h"(_fp8_h0_29));
                tr_vals[7] = _fp8_rt_7;
                {
                    uint16_t _fp8_pair_30;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_30) : "f"(0.0f), "f"(score_values[8]));
                    uint32_t _byte_30 = (uint32_t)(_fp8_pair_30 & 0xFF);
                    uint32_t _addr_30 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1024 + row ^ (1024 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_30), "r"(_byte_30) : "memory");
                }
                float _fp8_rt_8;
                uint16_t _e4m3x2_31;
                uint32_t _f16x2_31;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_31) : "f"(0.0f), "f"(score_values[8]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_31) : "h"(_e4m3x2_31));
                uint16_t _fp8_h0_31 = (uint16_t)(_f16x2_31 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_8) : "h"(_fp8_h0_31));
                tr_vals[8] = _fp8_rt_8;
                {
                    uint16_t _fp8_pair_32;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_32) : "f"(0.0f), "f"(score_values[9]));
                    uint32_t _byte_32 = (uint32_t)(_fp8_pair_32 & 0xFF);
                    uint32_t _addr_32 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1152 + row ^ (1152 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_32), "r"(_byte_32) : "memory");
                }
                float _fp8_rt_9;
                uint16_t _e4m3x2_33;
                uint32_t _f16x2_33;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_33) : "f"(0.0f), "f"(score_values[9]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_33) : "h"(_e4m3x2_33));
                uint16_t _fp8_h0_33 = (uint16_t)(_f16x2_33 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_9) : "h"(_fp8_h0_33));
                tr_vals[9] = _fp8_rt_9;
                {
                    uint16_t _fp8_pair_34;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_34) : "f"(0.0f), "f"(score_values[10]));
                    uint32_t _byte_34 = (uint32_t)(_fp8_pair_34 & 0xFF);
                    uint32_t _addr_34 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1280 + row ^ (1280 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_34), "r"(_byte_34) : "memory");
                }
                float _fp8_rt_10;
                uint16_t _e4m3x2_35;
                uint32_t _f16x2_35;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_35) : "f"(0.0f), "f"(score_values[10]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_35) : "h"(_e4m3x2_35));
                uint16_t _fp8_h0_35 = (uint16_t)(_f16x2_35 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_10) : "h"(_fp8_h0_35));
                tr_vals[10] = _fp8_rt_10;
                {
                    uint16_t _fp8_pair_36;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_36) : "f"(0.0f), "f"(score_values[11]));
                    uint32_t _byte_36 = (uint32_t)(_fp8_pair_36 & 0xFF);
                    uint32_t _addr_36 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1408 + row ^ (1408 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_36), "r"(_byte_36) : "memory");
                }
                float _fp8_rt_11;
                uint16_t _e4m3x2_37;
                uint32_t _f16x2_37;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_37) : "f"(0.0f), "f"(score_values[11]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_37) : "h"(_e4m3x2_37));
                uint16_t _fp8_h0_37 = (uint16_t)(_f16x2_37 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_11) : "h"(_fp8_h0_37));
                tr_vals[11] = _fp8_rt_11;
                {
                    uint16_t _fp8_pair_38;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_38) : "f"(0.0f), "f"(score_values[12]));
                    uint32_t _byte_38 = (uint32_t)(_fp8_pair_38 & 0xFF);
                    uint32_t _addr_38 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1536 + row ^ (1536 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_38), "r"(_byte_38) : "memory");
                }
                float _fp8_rt_12;
                uint16_t _e4m3x2_39;
                uint32_t _f16x2_39;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_39) : "f"(0.0f), "f"(score_values[12]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_39) : "h"(_e4m3x2_39));
                uint16_t _fp8_h0_39 = (uint16_t)(_f16x2_39 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_12) : "h"(_fp8_h0_39));
                tr_vals[12] = _fp8_rt_12;
                {
                    uint16_t _fp8_pair_40;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_40) : "f"(0.0f), "f"(score_values[13]));
                    uint32_t _byte_40 = (uint32_t)(_fp8_pair_40 & 0xFF);
                    uint32_t _addr_40 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1664 + row ^ (1664 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_40), "r"(_byte_40) : "memory");
                }
                float _fp8_rt_13;
                uint16_t _e4m3x2_41;
                uint32_t _f16x2_41;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_41) : "f"(0.0f), "f"(score_values[13]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_41) : "h"(_e4m3x2_41));
                uint16_t _fp8_h0_41 = (uint16_t)(_f16x2_41 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_13) : "h"(_fp8_h0_41));
                tr_vals[13] = _fp8_rt_13;
                {
                    uint16_t _fp8_pair_42;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_42) : "f"(0.0f), "f"(score_values[14]));
                    uint32_t _byte_42 = (uint32_t)(_fp8_pair_42 & 0xFF);
                    uint32_t _addr_42 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1792 + row ^ (1792 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_42), "r"(_byte_42) : "memory");
                }
                float _fp8_rt_14;
                uint16_t _e4m3x2_43;
                uint32_t _f16x2_43;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_43) : "f"(0.0f), "f"(score_values[14]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_43) : "h"(_e4m3x2_43));
                uint16_t _fp8_h0_43 = (uint16_t)(_f16x2_43 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_14) : "h"(_fp8_h0_43));
                tr_vals[14] = _fp8_rt_14;
                {
                    uint16_t _fp8_pair_44;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_44) : "f"(0.0f), "f"(score_values[15]));
                    uint32_t _byte_44 = (uint32_t)(_fp8_pair_44 & 0xFF);
                    uint32_t _addr_44 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1920 + row ^ (1920 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_44), "r"(_byte_44) : "memory");
                }
                float _fp8_rt_15;
                uint16_t _e4m3x2_45;
                uint32_t _f16x2_45;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_45) : "f"(0.0f), "f"(score_values[15]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_45) : "h"(_e4m3x2_45));
                uint16_t _fp8_h0_45 = (uint16_t)(_f16x2_45 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_15) : "h"(_fp8_h0_45));
                tr_vals[15] = _fp8_rt_15;
                int hi_bit_45 = lane & 16;
                float send_46 = ((hi_bit_45 != 0) ? score_values[0] : score_values[8]);
                float keep_47 = ((hi_bit_45 != 0) ? score_values[8] : score_values[0]);
                float _shfl_32 = __shfl_sync(0xFFFFFFFF, send_46, lane ^ 16);
                float recv_48 = _shfl_32;
                score_values[0] = keep_47 + recv_48;
                float send_49 = ((hi_bit_45 != 0) ? score_values[1] : score_values[9]);
                float keep_50 = ((hi_bit_45 != 0) ? score_values[9] : score_values[1]);
                float _shfl_33 = __shfl_sync(0xFFFFFFFF, send_49, lane ^ 16);
                float recv_51 = _shfl_33;
                score_values[1] = keep_50 + recv_51;
                float send_52 = ((hi_bit_45 != 0) ? score_values[2] : score_values[10]);
                float keep_53 = ((hi_bit_45 != 0) ? score_values[10] : score_values[2]);
                float _shfl_34 = __shfl_sync(0xFFFFFFFF, send_52, lane ^ 16);
                float recv_54 = _shfl_34;
                score_values[2] = keep_53 + recv_54;
                float send_55 = ((hi_bit_45 != 0) ? score_values[3] : score_values[11]);
                float keep_56 = ((hi_bit_45 != 0) ? score_values[11] : score_values[3]);
                float _shfl_35 = __shfl_sync(0xFFFFFFFF, send_55, lane ^ 16);
                float recv_57 = _shfl_35;
                score_values[3] = keep_56 + recv_57;
                float send_58 = ((hi_bit_45 != 0) ? score_values[4] : score_values[12]);
                float keep_59 = ((hi_bit_45 != 0) ? score_values[12] : score_values[4]);
                float _shfl_36 = __shfl_sync(0xFFFFFFFF, send_58, lane ^ 16);
                float recv_60 = _shfl_36;
                score_values[4] = keep_59 + recv_60;
                float send_61 = ((hi_bit_45 != 0) ? score_values[5] : score_values[13]);
                float keep_62 = ((hi_bit_45 != 0) ? score_values[13] : score_values[5]);
                float _shfl_37 = __shfl_sync(0xFFFFFFFF, send_61, lane ^ 16);
                float recv_63 = _shfl_37;
                score_values[5] = keep_62 + recv_63;
                float send_64 = ((hi_bit_45 != 0) ? score_values[6] : score_values[14]);
                float keep_65 = ((hi_bit_45 != 0) ? score_values[14] : score_values[6]);
                float _shfl_38 = __shfl_sync(0xFFFFFFFF, send_64, lane ^ 16);
                float recv_66 = _shfl_38;
                score_values[6] = keep_65 + recv_66;
                float send_67 = ((hi_bit_45 != 0) ? score_values[7] : score_values[15]);
                float keep_68 = ((hi_bit_45 != 0) ? score_values[15] : score_values[7]);
                float _shfl_39 = __shfl_sync(0xFFFFFFFF, send_67, lane ^ 16);
                float recv_69 = _shfl_39;
                score_values[7] = keep_68 + recv_69;
                int hi_bit_70 = lane & 8;
                float send_71 = ((hi_bit_70 != 0) ? score_values[0] : score_values[4]);
                float keep_72 = ((hi_bit_70 != 0) ? score_values[4] : score_values[0]);
                float _shfl_40 = __shfl_sync(0xFFFFFFFF, send_71, lane ^ 8);
                float recv_73 = _shfl_40;
                score_values[0] = keep_72 + recv_73;
                float send_74 = ((hi_bit_70 != 0) ? score_values[1] : score_values[5]);
                float keep_75 = ((hi_bit_70 != 0) ? score_values[5] : score_values[1]);
                float _shfl_41 = __shfl_sync(0xFFFFFFFF, send_74, lane ^ 8);
                float recv_76 = _shfl_41;
                score_values[1] = keep_75 + recv_76;
                float send_77 = ((hi_bit_70 != 0) ? score_values[2] : score_values[6]);
                float keep_78 = ((hi_bit_70 != 0) ? score_values[6] : score_values[2]);
                float _shfl_42 = __shfl_sync(0xFFFFFFFF, send_77, lane ^ 8);
                float recv_79 = _shfl_42;
                score_values[2] = keep_78 + recv_79;
                float send_80 = ((hi_bit_70 != 0) ? score_values[3] : score_values[7]);
                float keep_81 = ((hi_bit_70 != 0) ? score_values[7] : score_values[3]);
                float _shfl_43 = __shfl_sync(0xFFFFFFFF, send_80, lane ^ 8);
                float recv_82 = _shfl_43;
                score_values[3] = keep_81 + recv_82;
                int hi_bit_83 = lane & 4;
                float send_84 = ((hi_bit_83 != 0) ? score_values[0] : score_values[2]);
                float keep_85 = ((hi_bit_83 != 0) ? score_values[2] : score_values[0]);
                float _shfl_44 = __shfl_sync(0xFFFFFFFF, send_84, lane ^ 4);
                float recv_86 = _shfl_44;
                score_values[0] = keep_85 + recv_86;
                float send_87 = ((hi_bit_83 != 0) ? score_values[1] : score_values[3]);
                float keep_88 = ((hi_bit_83 != 0) ? score_values[3] : score_values[1]);
                float _shfl_45 = __shfl_sync(0xFFFFFFFF, send_87, lane ^ 4);
                float recv_89 = _shfl_45;
                score_values[1] = keep_88 + recv_89;
                int hi_bit_90 = lane & 2;
                float send_91 = ((hi_bit_90 != 0) ? score_values[0] : score_values[1]);
                float keep_92 = ((hi_bit_90 != 0) ? score_values[1] : score_values[0]);
                float _shfl_46 = __shfl_sync(0xFFFFFFFF, send_91, lane ^ 2);
                float recv_93 = _shfl_46;
                score_values[0] = keep_92 + recv_93;
                float _shfl_47 = __shfl_sync(0xFFFFFFFF, score_values[0], lane ^ 1);
                float other_94 = _shfl_47;
                score_values[0] = score_values[0] + other_94;
                int hi_bit_95 = lane & 16;
                float send_96 = ((hi_bit_95 != 0) ? tr_vals[0] : tr_vals[8]);
                float keep_97 = ((hi_bit_95 != 0) ? tr_vals[8] : tr_vals[0]);
                float _shfl_48 = __shfl_sync(0xFFFFFFFF, send_96, lane ^ 16);
                float recv_98 = _shfl_48;
                tr_vals[0] = keep_97 + recv_98;
                float send_99 = ((hi_bit_95 != 0) ? tr_vals[1] : tr_vals[9]);
                float keep_100 = ((hi_bit_95 != 0) ? tr_vals[9] : tr_vals[1]);
                float _shfl_49 = __shfl_sync(0xFFFFFFFF, send_99, lane ^ 16);
                float recv_101 = _shfl_49;
                tr_vals[1] = keep_100 + recv_101;
                float send_102 = ((hi_bit_95 != 0) ? tr_vals[2] : tr_vals[10]);
                float keep_103 = ((hi_bit_95 != 0) ? tr_vals[10] : tr_vals[2]);
                float _shfl_50 = __shfl_sync(0xFFFFFFFF, send_102, lane ^ 16);
                float recv_104 = _shfl_50;
                tr_vals[2] = keep_103 + recv_104;
                float send_105 = ((hi_bit_95 != 0) ? tr_vals[3] : tr_vals[11]);
                float keep_106 = ((hi_bit_95 != 0) ? tr_vals[11] : tr_vals[3]);
                float _shfl_51 = __shfl_sync(0xFFFFFFFF, send_105, lane ^ 16);
                float recv_107 = _shfl_51;
                tr_vals[3] = keep_106 + recv_107;
                float send_108 = ((hi_bit_95 != 0) ? tr_vals[4] : tr_vals[12]);
                float keep_109 = ((hi_bit_95 != 0) ? tr_vals[12] : tr_vals[4]);
                float _shfl_52 = __shfl_sync(0xFFFFFFFF, send_108, lane ^ 16);
                float recv_110 = _shfl_52;
                tr_vals[4] = keep_109 + recv_110;
                float send_111 = ((hi_bit_95 != 0) ? tr_vals[5] : tr_vals[13]);
                float keep_112 = ((hi_bit_95 != 0) ? tr_vals[13] : tr_vals[5]);
                float _shfl_53 = __shfl_sync(0xFFFFFFFF, send_111, lane ^ 16);
                float recv_113 = _shfl_53;
                tr_vals[5] = keep_112 + recv_113;
                float send_114 = ((hi_bit_95 != 0) ? tr_vals[6] : tr_vals[14]);
                float keep_115 = ((hi_bit_95 != 0) ? tr_vals[14] : tr_vals[6]);
                float _shfl_54 = __shfl_sync(0xFFFFFFFF, send_114, lane ^ 16);
                float recv_116 = _shfl_54;
                tr_vals[6] = keep_115 + recv_116;
                float send_117 = ((hi_bit_95 != 0) ? tr_vals[7] : tr_vals[15]);
                float keep_118 = ((hi_bit_95 != 0) ? tr_vals[15] : tr_vals[7]);
                float _shfl_55 = __shfl_sync(0xFFFFFFFF, send_117, lane ^ 16);
                float recv_119 = _shfl_55;
                tr_vals[7] = keep_118 + recv_119;
                int hi_bit_120 = lane & 8;
                float send_121 = ((hi_bit_120 != 0) ? tr_vals[0] : tr_vals[4]);
                float keep_122 = ((hi_bit_120 != 0) ? tr_vals[4] : tr_vals[0]);
                float _shfl_56 = __shfl_sync(0xFFFFFFFF, send_121, lane ^ 8);
                float recv_123 = _shfl_56;
                tr_vals[0] = keep_122 + recv_123;
                float send_124 = ((hi_bit_120 != 0) ? tr_vals[1] : tr_vals[5]);
                float keep_125 = ((hi_bit_120 != 0) ? tr_vals[5] : tr_vals[1]);
                float _shfl_57 = __shfl_sync(0xFFFFFFFF, send_124, lane ^ 8);
                float recv_126 = _shfl_57;
                tr_vals[1] = keep_125 + recv_126;
                float send_127 = ((hi_bit_120 != 0) ? tr_vals[2] : tr_vals[6]);
                float keep_128 = ((hi_bit_120 != 0) ? tr_vals[6] : tr_vals[2]);
                float _shfl_58 = __shfl_sync(0xFFFFFFFF, send_127, lane ^ 8);
                float recv_129 = _shfl_58;
                tr_vals[2] = keep_128 + recv_129;
                float send_130 = ((hi_bit_120 != 0) ? tr_vals[3] : tr_vals[7]);
                float keep_131 = ((hi_bit_120 != 0) ? tr_vals[7] : tr_vals[3]);
                float _shfl_59 = __shfl_sync(0xFFFFFFFF, send_130, lane ^ 8);
                float recv_132 = _shfl_59;
                tr_vals[3] = keep_131 + recv_132;
                int hi_bit_133 = lane & 4;
                float send_134 = ((hi_bit_133 != 0) ? tr_vals[0] : tr_vals[2]);
                float keep_135 = ((hi_bit_133 != 0) ? tr_vals[2] : tr_vals[0]);
                float _shfl_60 = __shfl_sync(0xFFFFFFFF, send_134, lane ^ 4);
                float recv_136 = _shfl_60;
                tr_vals[0] = keep_135 + recv_136;
                float send_137 = ((hi_bit_133 != 0) ? tr_vals[1] : tr_vals[3]);
                float keep_138 = ((hi_bit_133 != 0) ? tr_vals[3] : tr_vals[1]);
                float _shfl_61 = __shfl_sync(0xFFFFFFFF, send_137, lane ^ 4);
                float recv_139 = _shfl_61;
                tr_vals[1] = keep_138 + recv_139;
                int hi_bit_140 = lane & 2;
                float send_141 = ((hi_bit_140 != 0) ? tr_vals[0] : tr_vals[1]);
                float keep_142 = ((hi_bit_140 != 0) ? tr_vals[1] : tr_vals[0]);
                float _shfl_62 = __shfl_sync(0xFFFFFFFF, send_141, lane ^ 2);
                float recv_143 = _shfl_62;
                tr_vals[0] = keep_142 + recv_143;
                float _shfl_63 = __shfl_sync(0xFFFFFFFF, tr_vals[0], lane ^ 1);
                float other_144 = _shfl_63;
                tr_vals[0] = tr_vals[0] + other_144;
                if ((lane & 1) == 0) {
                    smem_psum[local_warp * 16 + (lane >> 1)] = score_values[0];
                    smem_rsum[local_warp * 16 + (lane >> 1)] = tr_vals[0];
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                float norm_lane = 0.0f;
                if (lane < 16) {
                    float _exp2_16 = approx_exp2(sink_lane - m_scaled);
                    float sink_term = _exp2_16;
                    float col_sum = smem_psum[lane] + smem_psum[16 + lane] + smem_psum[32 + lane] + smem_psum[48 + lane] + sink_term;
                    float denom = smem_rsum[lane] + smem_rsum[16 + lane] + smem_rsum[32 + lane] + smem_rsum[48 + lane] + sink_term;
                    if (denom > 0.0f) {
                        float _rcp_0 = approx_rcp(denom);
                        norm_lane = _rcp_0 * output_scale;
                    }
                    if (local_warp == 0) {
                        if (o_chunk == 0 && head_base + lane < num_heads) {
                            int lse_offset = (query_idx * num_heads + head_base + lane) * num_splits + split_idx;
                            float _log2_0;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(col_sum));
                            partial_lse[lse_offset] = ((col_sum > 0.0f) ? (m_scaled + _log2_0) * lse_partial_scale : -CAKE_INF);
                        }
                    }
                }
                float norm_c[16];
                float _shfl_64 = __shfl_sync(0xFFFFFFFF, norm_lane, 0);
                norm_c[0] = _shfl_64;
                float _shfl_65 = __shfl_sync(0xFFFFFFFF, norm_lane, 1);
                norm_c[1] = _shfl_65;
                float _shfl_66 = __shfl_sync(0xFFFFFFFF, norm_lane, 2);
                norm_c[2] = _shfl_66;
                float _shfl_67 = __shfl_sync(0xFFFFFFFF, norm_lane, 3);
                norm_c[3] = _shfl_67;
                float _shfl_68 = __shfl_sync(0xFFFFFFFF, norm_lane, 4);
                norm_c[4] = _shfl_68;
                float _shfl_69 = __shfl_sync(0xFFFFFFFF, norm_lane, 5);
                norm_c[5] = _shfl_69;
                float _shfl_70 = __shfl_sync(0xFFFFFFFF, norm_lane, 6);
                norm_c[6] = _shfl_70;
                float _shfl_71 = __shfl_sync(0xFFFFFFFF, norm_lane, 7);
                norm_c[7] = _shfl_71;
                float _shfl_72 = __shfl_sync(0xFFFFFFFF, norm_lane, 8);
                norm_c[8] = _shfl_72;
                float _shfl_73 = __shfl_sync(0xFFFFFFFF, norm_lane, 9);
                norm_c[9] = _shfl_73;
                float _shfl_74 = __shfl_sync(0xFFFFFFFF, norm_lane, 10);
                norm_c[10] = _shfl_74;
                float _shfl_75 = __shfl_sync(0xFFFFFFFF, norm_lane, 11);
                norm_c[11] = _shfl_75;
                float _shfl_76 = __shfl_sync(0xFFFFFFFF, norm_lane, 12);
                norm_c[12] = _shfl_76;
                float _shfl_77 = __shfl_sync(0xFFFFFFFF, norm_lane, 13);
                norm_c[13] = _shfl_77;
                float _shfl_78 = __shfl_sync(0xFFFFFFFF, norm_lane, 14);
                norm_c[14] = _shfl_78;
                float _shfl_79 = __shfl_sync(0xFFFFFFFF, norm_lane, 15);
                norm_c[15] = _shfl_79;
                float o_values[16];
                int dim = 0;
                long long out_off = 0;
                mbarrier_wait_hint(o_full_addr, _phase_o_full_0, 10000000);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x16(&o_values[0], taddr + 32 + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim = o_chunk * 128 + row;
                if (head_base < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[0] * norm_c[0];
                }
                if (head_base + 1 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 1) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[1] * norm_c[1];
                }
                if (head_base + 2 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 2) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[2] * norm_c[2];
                }
                if (head_base + 3 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 3) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[3] * norm_c[3];
                }
                if (head_base + 4 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 4) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[4] * norm_c[4];
                }
                if (head_base + 5 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 5) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[5] * norm_c[5];
                }
                if (head_base + 6 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 6) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[6] * norm_c[6];
                }
                if (head_base + 7 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 7) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[7] * norm_c[7];
                }
                if (head_base + 8 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 8) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[8] * norm_c[8];
                }
                if (head_base + 9 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 9) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[9] * norm_c[9];
                }
                if (head_base + 10 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 10) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[10] * norm_c[10];
                }
                if (head_base + 11 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 11) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[11] * norm_c[11];
                }
                if (head_base + 12 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 12) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[12] * norm_c[12];
                }
                if (head_base + 13 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 13) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[13] * norm_c[13];
                }
                if (head_base + 14 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 14) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[14] * norm_c[14];
                }
                if (head_base + 15 < num_heads) {
                    out_off = ((long long)(query_idx * num_heads + head_base + 15) * (long long)num_splits + (long long)split_idx) * 512 + (long long)dim;
                    partial_O[out_off] = o_values[15] * norm_c[15];
                }
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }
    // ---- Role: compute1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // compute1_main
            const int local_warp_1 = warp - 4;
            int o_chunk_1 = blockIdx.x % 4;
            int work_idx_1 = blockIdx.x / 4;
            int head_tile_1 = work_idx_1 % num_head_tiles;
            int split_work_1 = work_idx_1 / num_head_tiles;
            int split_idx_1 = split_work_1 % num_splits;
            int query_idx_1 = split_work_1 / num_splits;
            int head_base_1 = head_tile_1 * 128;
            const int row_1 = local_warp_1 * 32 + lane;
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
            int strip_1 = smem_sfs_addr + (unsigned int)(row_1 * 32);
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            const int gather_tid_1 = warp * 32 + lane;
            const int g_chunk_1 = gather_tid_1 % 24;
            const int g_row0_1 = gather_tid_1 / 24;
            const int g_kind_1 = ((g_chunk_1 < 14) ? 0 : ((g_chunk_1 < 22) ? 1 : 2));
            long long g_src_off_1 = (long long)(((g_kind_1 < 2) ? 16 * g_chunk_1 : 16 * (g_chunk_1 - 14 - 8)));
            int g_dst_k_1 = smem_kf4_addr + (unsigned int)(g_chunk_1 / 8 * 16384) + (unsigned int)(g_row0_1 * 128 + (g_chunk_1 % 8 * 16 ^ g_row0_1 % 8 * 16));
            int g_dst_r_1 = smem_krope_addr + (unsigned int)(g_row0_1 * 128 + ((g_chunk_1 - 14) * 16 ^ g_row0_1 % 8 * 16));
            int g_dst_s_1 = smem_sfs_addr + (unsigned int)(g_row0_1 * 32) + (unsigned int)(16 * (g_chunk_1 - 14 - 8));
            int g_dst0_1 = ((g_kind_1 == 0) ? g_dst_k_1 : ((g_kind_1 == 1) ? g_dst_r_1 : g_dst_s_1));
            const int g_row_bytes_1 = ((g_kind_1 < 2) ? 128 : 32);
            unsigned int w4_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)(g_row0_1 * 16)));
            long long od_2 = (long long)w4_1[1] << 32 | (long long)w4_1[0];
            long long osf_1 = (long long)w4_1[3] << 32 | (long long)w4_1[2];
            long long goff_1 = ((g_kind_1 < 2) ? od_2 : osf_1);
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
            long long goff_3_1 = ((g_kind_1 < 2) ? od_1_1 : osf_2_1);
            if (goff_3_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 16 * g_row_bytes_1), "l"(cache_1 + (goff_3_1 + g_src_off_1)));
            }
            unsigned int w4_4_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 32) * 16)));
            long long od_5_1 = (long long)w4_4_1[1] << 32 | (long long)w4_4_1[0];
            long long osf_6_1 = (long long)w4_4_1[3] << 32 | (long long)w4_4_1[2];
            long long goff_7_1 = ((g_kind_1 < 2) ? od_5_1 : osf_6_1);
            if (goff_7_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 32 * g_row_bytes_1), "l"(cache_1 + (goff_7_1 + g_src_off_1)));
            }
            unsigned int w4_8_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_8_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 48) * 16)));
            long long od_9_1 = (long long)w4_8_1[1] << 32 | (long long)w4_8_1[0];
            long long osf_10_1 = (long long)w4_8_1[3] << 32 | (long long)w4_8_1[2];
            long long goff_11_1 = ((g_kind_1 < 2) ? od_9_1 : osf_10_1);
            if (goff_11_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 48 * g_row_bytes_1), "l"(cache_1 + (goff_11_1 + g_src_off_1)));
            }
            unsigned int w4_12_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 64) * 16)));
            long long od_13_1 = (long long)w4_12_1[1] << 32 | (long long)w4_12_1[0];
            long long osf_14_1 = (long long)w4_12_1[3] << 32 | (long long)w4_12_1[2];
            long long goff_15_1 = ((g_kind_1 < 2) ? od_13_1 : osf_14_1);
            if (goff_15_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 64 * g_row_bytes_1), "l"(cache_1 + (goff_15_1 + g_src_off_1)));
            }
            unsigned int w4_16_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_16_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 80) * 16)));
            long long od_17_1 = (long long)w4_16_1[1] << 32 | (long long)w4_16_1[0];
            long long osf_18_1 = (long long)w4_16_1[3] << 32 | (long long)w4_16_1[2];
            long long goff_19_1 = ((g_kind_1 < 2) ? od_17_1 : osf_18_1);
            if (goff_19_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 80 * g_row_bytes_1), "l"(cache_1 + (goff_19_1 + g_src_off_1)));
            }
            unsigned int w4_20_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_20_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 96) * 16)));
            long long od_21_1 = (long long)w4_20_1[1] << 32 | (long long)w4_20_1[0];
            long long osf_22_1 = (long long)w4_20_1[3] << 32 | (long long)w4_20_1[2];
            long long goff_23_1 = ((g_kind_1 < 2) ? od_21_1 : osf_22_1);
            if (goff_23_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 96 * g_row_bytes_1), "l"(cache_1 + (goff_23_1 + g_src_off_1)));
            }
            unsigned int w4_24_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_24_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_1[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_1 + 112) * 16)));
            long long od_25_1 = (long long)w4_24_1[1] << 32 | (long long)w4_24_1[0];
            long long osf_26_1 = (long long)w4_24_1[3] << 32 | (long long)w4_24_1[2];
            long long goff_27_1 = ((g_kind_1 < 2) ? od_25_1 : osf_26_1);
            if (goff_27_1 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_1 + 112 * g_row_bytes_1), "l"(cache_1 + (goff_27_1 + g_src_off_1)));
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
                    if (head_base_1 + q_block_1 * 32 < num_heads && q_row_1 < 32) {
                        int q_row_addr_1 = smem_qstage_addr + (unsigned int)(kset_1 * 4096) + (unsigned int)(q_row_1 * 128);
                        unsigned int sf_word_1 = 0;
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
                            float _max_50 = max_noftz(_fabs_32, _fabs_33);
                            m8_1[0] = _max_50;
                            float _fabs_34 = fabsf(qv_1[2]);
                            float _fabs_35 = fabsf(qv_1[3]);
                            float _max_51 = max_noftz(_fabs_34, _fabs_35);
                            m8_1[1] = _max_51;
                            float _fabs_36 = fabsf(qv_1[4]);
                            float _fabs_37 = fabsf(qv_1[5]);
                            float _max_52 = max_noftz(_fabs_36, _fabs_37);
                            m8_1[2] = _max_52;
                            float _fabs_38 = fabsf(qv_1[6]);
                            float _fabs_39 = fabsf(qv_1[7]);
                            float _max_53 = max_noftz(_fabs_38, _fabs_39);
                            m8_1[3] = _max_53;
                            float _fabs_40 = fabsf(qv_1[8]);
                            float _fabs_41 = fabsf(qv_1[9]);
                            float _max_54 = max_noftz(_fabs_40, _fabs_41);
                            m8_1[4] = _max_54;
                            float _fabs_42 = fabsf(qv_1[10]);
                            float _fabs_43 = fabsf(qv_1[11]);
                            float _max_55 = max_noftz(_fabs_42, _fabs_43);
                            m8_1[5] = _max_55;
                            float _fabs_44 = fabsf(qv_1[12]);
                            float _fabs_45 = fabsf(qv_1[13]);
                            float _max_56 = max_noftz(_fabs_44, _fabs_45);
                            m8_1[6] = _max_56;
                            float _fabs_46 = fabsf(qv_1[14]);
                            float _fabs_47 = fabsf(qv_1[15]);
                            float _max_57 = max_noftz(_fabs_46, _fabs_47);
                            m8_1[7] = _max_57;
                            float m4_1[4];
                            float _max_58 = max_noftz(m8_1[0], m8_1[1]);
                            m4_1[0] = _max_58;
                            float _max_59 = max_noftz(m8_1[2], m8_1[3]);
                            m4_1[1] = _max_59;
                            float _max_60 = max_noftz(m8_1[4], m8_1[5]);
                            m4_1[2] = _max_60;
                            float _max_61 = max_noftz(m8_1[6], m8_1[7]);
                            m4_1[3] = _max_61;
                            float _max_62 = max_noftz(m4_1[0], m4_1[1]);
                            float _max_63 = max_noftz(m4_1[2], m4_1[3]);
                            float _max_64 = max_noftz(_max_62, _max_63);
                            float amax_1 = _max_64;
                            float sc_1 = amax_1 * inv_six_1;
                            uint16_t _e4m3x2_f32_50;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_50) : "f"(0.0f), "f"(sc_1));
                            uint16_t sc_pair_1 = _e4m3x2_f32_50;
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
                            sf_word_1 = sf_word_1 | sc_byte_1 << (unsigned int)(8 * (2 * bp_1));
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
                            float _max_65 = max_noftz(_fabs_48, _fabs_49);
                            m8_3_1[0] = _max_65;
                            float _fabs_50 = fabsf(qv_2_1[2]);
                            float _fabs_51 = fabsf(qv_2_1[3]);
                            float _max_66 = max_noftz(_fabs_50, _fabs_51);
                            m8_3_1[1] = _max_66;
                            float _fabs_52 = fabsf(qv_2_1[4]);
                            float _fabs_53 = fabsf(qv_2_1[5]);
                            float _max_67 = max_noftz(_fabs_52, _fabs_53);
                            m8_3_1[2] = _max_67;
                            float _fabs_54 = fabsf(qv_2_1[6]);
                            float _fabs_55 = fabsf(qv_2_1[7]);
                            float _max_68 = max_noftz(_fabs_54, _fabs_55);
                            m8_3_1[3] = _max_68;
                            float _fabs_56 = fabsf(qv_2_1[8]);
                            float _fabs_57 = fabsf(qv_2_1[9]);
                            float _max_69 = max_noftz(_fabs_56, _fabs_57);
                            m8_3_1[4] = _max_69;
                            float _fabs_58 = fabsf(qv_2_1[10]);
                            float _fabs_59 = fabsf(qv_2_1[11]);
                            float _max_70 = max_noftz(_fabs_58, _fabs_59);
                            m8_3_1[5] = _max_70;
                            float _fabs_60 = fabsf(qv_2_1[12]);
                            float _fabs_61 = fabsf(qv_2_1[13]);
                            float _max_71 = max_noftz(_fabs_60, _fabs_61);
                            m8_3_1[6] = _max_71;
                            float _fabs_62 = fabsf(qv_2_1[14]);
                            float _fabs_63 = fabsf(qv_2_1[15]);
                            float _max_72 = max_noftz(_fabs_62, _fabs_63);
                            m8_3_1[7] = _max_72;
                            float m4_4_1[4];
                            float _max_73 = max_noftz(m8_3_1[0], m8_3_1[1]);
                            m4_4_1[0] = _max_73;
                            float _max_74 = max_noftz(m8_3_1[2], m8_3_1[3]);
                            m4_4_1[1] = _max_74;
                            float _max_75 = max_noftz(m8_3_1[4], m8_3_1[5]);
                            m4_4_1[2] = _max_75;
                            float _max_76 = max_noftz(m8_3_1[6], m8_3_1[7]);
                            m4_4_1[3] = _max_76;
                            float _max_77 = max_noftz(m4_4_1[0], m4_4_1[1]);
                            float _max_78 = max_noftz(m4_4_1[2], m4_4_1[3]);
                            float _max_79 = max_noftz(_max_77, _max_78);
                            float amax_5_1 = _max_79;
                            float sc_6_1 = amax_5_1 * inv_six_1;
                            uint16_t _e4m3x2_f32_51;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_51) : "f"(0.0f), "f"(sc_6_1));
                            uint16_t sc_pair_7_1 = _e4m3x2_f32_51;
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
                            sf_word_1 = sf_word_1 | sc_byte_8_1 << (unsigned int)(8 * (2 * bp_1 + 1));
                            int chunk_1 = 2 * kset_1 + bp_1;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_qf4_addr + (unsigned int)(chunk_1 / 8 * 16384 + (q_row_1 * 128 + (chunk_1 % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3])));
                        }
                        smem_qsf32[kset_1 / 4 * 2048 + q_row_1 % 32 / 8 * 512 + kset_1 % 4 * 128 + q_row_1 % 8 * 16 + q_row_1 / 32 % 4 * 4 >> 2] = sf_word_1;
                    } else {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)(2 * kset_1 / 8 * 16384 + (q_row_1 * 128 + (2 * kset_1 % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_qf4_addr + (unsigned int)((2 * kset_1 + 1) / 8 * 16384 + (q_row_1 * 128 + ((2 * kset_1 + 1) % 8 * 16 ^ q_row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        smem_qsf32[kset_1 / 4 * 2048 + q_row_1 % 32 / 8 * 512 + kset_1 % 4 * 128 + q_row_1 % 8 * 16 + q_row_1 / 32 % 4 * 4 >> 2] = 0;
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_ready_addr);
            asm volatile("cp.async.wait_group 0;");
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            if (valid_1 != 0) {
                {
                    unsigned int sfw_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sfw_1[(0) + 3]))
                        : "r"(strip_1 + 16));
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_1[0];
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_1[1];
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = sfw_1[2];
                    {
                        smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    }
                }
                int vblock_1 = 8 * o_chunk_1 + 3;
                unsigned int v8_2[4];
                {
                    int vchunk_3 = vblock_1 >> 1;
                    int vhalf_3 = vblock_1 & 1;
                    unsigned int kraw_3[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_3[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_3 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_3 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_3)));
                    unsigned int sfw32_3 = smem_sfs32[row_1 * 32 + vblock_1 >> 2];
                    unsigned int scale_3 = sfw32_3 >> (unsigned int)(8 * (vblock_1 & 3)) & 255;
                    {
                        v8_2[0] = cake_dsv4_qmul4_portable<5>(kraw_3[0], scale_3);
                    }
                    {
                        v8_2[1] = cake_dsv4_qmul4_portable<6>(kraw_3[0], scale_3);
                    }
                    {
                        v8_2[2] = cake_dsv4_qmul4_portable<5>(kraw_3[1], scale_3);
                    }
                    {
                        v8_2[3] = cake_dsv4_qmul4_portable<6>(kraw_3[1], scale_3);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row_1 * 128 + (48 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 3])));
                int vblock_0_1 = 8 * o_chunk_1 + 4;
                unsigned int v8_1_1[4];
                if (vblock_0_1 < 28) {
                    int vchunk_4 = vblock_0_1 >> 1;
                    int vhalf_4 = vblock_0_1 & 1;
                    unsigned int kraw_4[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_4[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_4 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_4 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_4)));
                    unsigned int sfw32_4 = smem_sfs32[row_1 * 32 + vblock_0_1 >> 2];
                    unsigned int scale_4 = sfw32_4 >> (unsigned int)(8 * (vblock_0_1 & 3)) & 255;
                    {
                        v8_1_1[0] = cake_dsv4_qmul4_portable<5>(kraw_4[0], scale_4);
                    }
                    {
                        v8_1_1[1] = cake_dsv4_qmul4_portable<6>(kraw_4[0], scale_4);
                    }
                    {
                        v8_1_1[2] = cake_dsv4_qmul4_portable<5>(kraw_4[1], scale_4);
                    }
                    {
                        v8_1_1[3] = cake_dsv4_qmul4_portable<6>(kraw_4[1], scale_4);
                    }
                } else {
                    int rblock = vblock_0_1 - 28;
                    unsigned int rope[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_1 * 128 + (2 * rblock * 16 ^ row_1 % 8 * 16))));
                    float lo = __uint_as_float(rope[0] << 16);
                    float hi = __uint_as_float(rope[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_76;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_76) : "f"(hi), "f"(lo));
                    uint16_t pair = _e4m3x2_f32_76;
                    {
                        v8_1_1[0] = (unsigned int)pair;
                    }
                    float lo_0 = __uint_as_float(rope[1] << 16);
                    float hi_1 = __uint_as_float(rope[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_77;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_77) : "f"(hi_1), "f"(lo_0));
                    uint16_t pair_2 = _e4m3x2_f32_77;
                    {
                        v8_1_1[0] = v8_1_1[0] | (unsigned int)pair_2 << 16;
                    }
                    float lo_3 = __uint_as_float(rope[2] << 16);
                    float hi_4 = __uint_as_float(rope[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_78;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_78) : "f"(hi_4), "f"(lo_3));
                    uint16_t pair_5 = _e4m3x2_f32_78;
                    {
                        v8_1_1[1] = (unsigned int)pair_5;
                    }
                    float lo_6 = __uint_as_float(rope[3] << 16);
                    float hi_7 = __uint_as_float(rope[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_79;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_79) : "f"(hi_7), "f"(lo_6));
                    uint16_t pair_8 = _e4m3x2_f32_79;
                    {
                        v8_1_1[1] = v8_1_1[1] | (unsigned int)pair_8 << 16;
                    }
                    unsigned int rope_9[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_1 * 128 + ((2 * rblock + 1) * 16 ^ row_1 % 8 * 16))));
                    float lo_10 = __uint_as_float(rope_9[0] << 16);
                    float hi_11 = __uint_as_float(rope_9[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_80;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_80) : "f"(hi_11), "f"(lo_10));
                    uint16_t pair_12 = _e4m3x2_f32_80;
                    {
                        v8_1_1[2] = (unsigned int)pair_12;
                    }
                    float lo_13 = __uint_as_float(rope_9[1] << 16);
                    float hi_14 = __uint_as_float(rope_9[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_81;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_81) : "f"(hi_14), "f"(lo_13));
                    uint16_t pair_15 = _e4m3x2_f32_81;
                    {
                        v8_1_1[2] = v8_1_1[2] | (unsigned int)pair_15 << 16;
                    }
                    float lo_16 = __uint_as_float(rope_9[2] << 16);
                    float hi_17 = __uint_as_float(rope_9[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_82;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_82) : "f"(hi_17), "f"(lo_16));
                    uint16_t pair_18 = _e4m3x2_f32_82;
                    {
                        v8_1_1[3] = (unsigned int)pair_18;
                    }
                    float lo_19 = __uint_as_float(rope_9[3] << 16);
                    float hi_20 = __uint_as_float(rope_9[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_83;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_83) : "f"(hi_20), "f"(lo_19));
                    uint16_t pair_21 = _e4m3x2_f32_83;
                    {
                        v8_1_1[3] = v8_1_1[3] | (unsigned int)pair_21 << 16;
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row_1 * 128 + (64 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 3])));
                int vblock_2_1 = 8 * o_chunk_1 + 5;
                unsigned int v8_3_1[4];
                if (vblock_2_1 < 28) {
                    int vchunk_5 = vblock_2_1 >> 1;
                    int vhalf_5 = vblock_2_1 & 1;
                    unsigned int kraw_5[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_5[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_5 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_5 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_5)));
                    unsigned int sfw32_5 = smem_sfs32[row_1 * 32 + vblock_2_1 >> 2];
                    unsigned int scale_5 = sfw32_5 >> (unsigned int)(8 * (vblock_2_1 & 3)) & 255;
                    {
                        v8_3_1[0] = cake_dsv4_qmul4_portable<5>(kraw_5[0], scale_5);
                    }
                    {
                        v8_3_1[1] = cake_dsv4_qmul4_portable<6>(kraw_5[0], scale_5);
                    }
                    {
                        v8_3_1[2] = cake_dsv4_qmul4_portable<5>(kraw_5[1], scale_5);
                    }
                    {
                        v8_3_1[3] = cake_dsv4_qmul4_portable<6>(kraw_5[1], scale_5);
                    }
                } else {
                    int rblock_1 = vblock_2_1 - 28;
                    unsigned int rope_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_1 * 128 + (2 * rblock_1 * 16 ^ row_1 % 8 * 16))));
                    float lo_1 = __uint_as_float(rope_1[0] << 16);
                    float hi_2 = __uint_as_float(rope_1[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_92;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_92) : "f"(hi_2), "f"(lo_1));
                    uint16_t pair_1 = _e4m3x2_f32_92;
                    {
                        v8_3_1[0] = (unsigned int)pair_1;
                    }
                    float lo_0_1 = __uint_as_float(rope_1[1] << 16);
                    float hi_1_1 = __uint_as_float(rope_1[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_93;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_93) : "f"(hi_1_1), "f"(lo_0_1));
                    uint16_t pair_2_1 = _e4m3x2_f32_93;
                    {
                        v8_3_1[0] = v8_3_1[0] | (unsigned int)pair_2_1 << 16;
                    }
                    float lo_3_1 = __uint_as_float(rope_1[2] << 16);
                    float hi_4_1 = __uint_as_float(rope_1[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_94;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_94) : "f"(hi_4_1), "f"(lo_3_1));
                    uint16_t pair_5_1 = _e4m3x2_f32_94;
                    {
                        v8_3_1[1] = (unsigned int)pair_5_1;
                    }
                    float lo_6_1 = __uint_as_float(rope_1[3] << 16);
                    float hi_7_1 = __uint_as_float(rope_1[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_95;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_95) : "f"(hi_7_1), "f"(lo_6_1));
                    uint16_t pair_8_1 = _e4m3x2_f32_95;
                    {
                        v8_3_1[1] = v8_3_1[1] | (unsigned int)pair_8_1 << 16;
                    }
                    unsigned int rope_9_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_1 * 128 + ((2 * rblock_1 + 1) * 16 ^ row_1 % 8 * 16))));
                    float lo_10_1 = __uint_as_float(rope_9_1[0] << 16);
                    float hi_11_1 = __uint_as_float(rope_9_1[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_96;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_96) : "f"(hi_11_1), "f"(lo_10_1));
                    uint16_t pair_12_1 = _e4m3x2_f32_96;
                    {
                        v8_3_1[2] = (unsigned int)pair_12_1;
                    }
                    float lo_13_1 = __uint_as_float(rope_9_1[1] << 16);
                    float hi_14_1 = __uint_as_float(rope_9_1[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_97;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_97) : "f"(hi_14_1), "f"(lo_13_1));
                    uint16_t pair_15_1 = _e4m3x2_f32_97;
                    {
                        v8_3_1[2] = v8_3_1[2] | (unsigned int)pair_15_1 << 16;
                    }
                    float lo_16_1 = __uint_as_float(rope_9_1[2] << 16);
                    float hi_17_1 = __uint_as_float(rope_9_1[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_98;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_98) : "f"(hi_17_1), "f"(lo_16_1));
                    uint16_t pair_18_1 = _e4m3x2_f32_98;
                    {
                        v8_3_1[3] = (unsigned int)pair_18_1;
                    }
                    float lo_19_1 = __uint_as_float(rope_9_1[3] << 16);
                    float hi_20_1 = __uint_as_float(rope_9_1[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_99;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_99) : "f"(hi_20_1), "f"(lo_19_1));
                    uint16_t pair_21_1 = _e4m3x2_f32_99;
                    {
                        v8_3_1[3] = v8_3_1[3] | (unsigned int)pair_21_1 << 16;
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row_1 * 128 + (80 ^ row_1 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 3])));
            } else {
                {
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row_1 * 128 + (48 ^ row_1 % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row_1 * 128 + (64 ^ row_1 % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row_1 * 128 + (80 ^ row_1 % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_1 = bmm2_scale[0];
            unsigned int _phase_s_full_0_1 = 0;
            unsigned int _phase_o_full_0_1 = 0;
            {
                mbarrier_wait_hint(s_full_addr, _phase_s_full_0_1, 10000000);
                _phase_s_full_0_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float score_values_1[8];
                tmem_ld_x8(&score_values_1[0], taddr + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (valid_1 == 0) {
                    score_values_1[0] = -CAKE_INF;
                    score_values_1[1] = -CAKE_INF;
                    score_values_1[2] = -CAKE_INF;
                    score_values_1[3] = -CAKE_INF;
                    score_values_1[4] = -CAKE_INF;
                    score_values_1[5] = -CAKE_INF;
                    score_values_1[6] = -CAKE_INF;
                    score_values_1[7] = -CAKE_INF;
                }
                float tr_vals_1[8];
                tr_vals_1[0] = score_values_1[0];
                tr_vals_1[1] = score_values_1[1];
                tr_vals_1[2] = score_values_1[2];
                tr_vals_1[3] = score_values_1[3];
                tr_vals_1[4] = score_values_1[4];
                tr_vals_1[5] = score_values_1[5];
                tr_vals_1[6] = score_values_1[6];
                tr_vals_1[7] = score_values_1[7];
                int hi_bit_1 = lane & 16;
                float send_1 = ((hi_bit_1 != 0) ? tr_vals_1[0] : tr_vals_1[4]);
                float keep_2 = ((hi_bit_1 != 0) ? tr_vals_1[4] : tr_vals_1[0]);
                float _shfl_80 = __shfl_sync(0xFFFFFFFF, send_1, lane ^ 16);
                float recv_1 = _shfl_80;
                float _max_80 = max_noftz(keep_2, recv_1);
                tr_vals_1[0] = _max_80;
                float send_0_1 = ((hi_bit_1 != 0) ? tr_vals_1[1] : tr_vals_1[5]);
                float keep_1_1 = ((hi_bit_1 != 0) ? tr_vals_1[5] : tr_vals_1[1]);
                float _shfl_81 = __shfl_sync(0xFFFFFFFF, send_0_1, lane ^ 16);
                float recv_2_1 = _shfl_81;
                float _max_81 = max_noftz(keep_1_1, recv_2_1);
                tr_vals_1[1] = _max_81;
                float send_3_1 = ((hi_bit_1 != 0) ? tr_vals_1[2] : tr_vals_1[6]);
                float keep_4_1 = ((hi_bit_1 != 0) ? tr_vals_1[6] : tr_vals_1[2]);
                float _shfl_82 = __shfl_sync(0xFFFFFFFF, send_3_1, lane ^ 16);
                float recv_5_1 = _shfl_82;
                float _max_82 = max_noftz(keep_4_1, recv_5_1);
                tr_vals_1[2] = _max_82;
                float send_6_1 = ((hi_bit_1 != 0) ? tr_vals_1[3] : tr_vals_1[7]);
                float keep_7_1 = ((hi_bit_1 != 0) ? tr_vals_1[7] : tr_vals_1[3]);
                float _shfl_83 = __shfl_sync(0xFFFFFFFF, send_6_1, lane ^ 16);
                float recv_8_1 = _shfl_83;
                float _max_83 = max_noftz(keep_7_1, recv_8_1);
                tr_vals_1[3] = _max_83;
                int hi_bit_9 = lane & 8;
                float send_10 = ((hi_bit_9 != 0) ? tr_vals_1[0] : tr_vals_1[2]);
                float keep_11 = ((hi_bit_9 != 0) ? tr_vals_1[2] : tr_vals_1[0]);
                float _shfl_84 = __shfl_sync(0xFFFFFFFF, send_10, lane ^ 8);
                float recv_12 = _shfl_84;
                float _max_84 = max_noftz(keep_11, recv_12);
                tr_vals_1[0] = _max_84;
                float send_13 = ((hi_bit_9 != 0) ? tr_vals_1[1] : tr_vals_1[3]);
                float keep_14 = ((hi_bit_9 != 0) ? tr_vals_1[3] : tr_vals_1[1]);
                float _shfl_85 = __shfl_sync(0xFFFFFFFF, send_13, lane ^ 8);
                float recv_15 = _shfl_85;
                float _max_85 = max_noftz(keep_14, recv_15);
                tr_vals_1[1] = _max_85;
                int hi_bit_16 = lane & 4;
                float send_17 = ((hi_bit_16 != 0) ? tr_vals_1[0] : tr_vals_1[1]);
                float keep_18 = ((hi_bit_16 != 0) ? tr_vals_1[1] : tr_vals_1[0]);
                float _shfl_86 = __shfl_sync(0xFFFFFFFF, send_17, lane ^ 4);
                float recv_19 = _shfl_86;
                float _max_86 = max_noftz(keep_18, recv_19);
                tr_vals_1[0] = _max_86;
                float _shfl_87 = __shfl_sync(0xFFFFFFFF, tr_vals_1[0], lane ^ 2);
                float other_1 = _shfl_87;
                float _max_87 = max_noftz(tr_vals_1[0], other_1);
                tr_vals_1[0] = _max_87;
                float _shfl_88 = __shfl_sync(0xFFFFFFFF, tr_vals_1[0], lane ^ 1);
                float other_20 = _shfl_88;
                float _max_88 = max_noftz(tr_vals_1[0], other_20);
                tr_vals_1[0] = _max_88;
                if ((lane & 3) == 0) {
                    smem_pmax[(4 + local_warp_1) * 16 + (lane >> 2)] = tr_vals_1[0];
                }
                asm volatile("barrier.sync 11, 128;" ::: "memory");
                float m_lane_1 = -CAKE_INF;
                float sink_lane_1 = -CAKE_INF;
                float m_scaled_1 = 0.0f;
                if (lane < 8) {
                    float _max_89 = max_noftz(smem_pmax[64 + lane], smem_pmax[80 + lane]);
                    float _max_90 = max_noftz(smem_pmax[96 + lane], smem_pmax[112 + lane]);
                    float _max_91 = max_noftz(_max_89, _max_90);
                    m_lane_1 = _max_91;
                    if (has_sinks != 0 && split_idx_1 == 0 && head_base_1 + 16 + lane < num_heads) {
                        sink_lane_1 = sinks[head_base_1 + 16 + lane] * 1.4426950408889634f;
                    }
                    float _max_92 = max_noftz(m_lane_1 * softmax_scale_log2_1, sink_lane_1);
                    m_scaled_1 = _max_92;
                    if (m_scaled_1 == -CAKE_INF) {
                        m_scaled_1 = 0.0f;
                    }
                }
                float col_max_1[8];
                float _shfl_89 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 0);
                col_max_1[0] = _shfl_89;
                float _shfl_90 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 1);
                col_max_1[1] = _shfl_90;
                float _shfl_91 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 2);
                col_max_1[2] = _shfl_91;
                float _shfl_92 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 3);
                col_max_1[3] = _shfl_92;
                float _shfl_93 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 4);
                col_max_1[4] = _shfl_93;
                float _shfl_94 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 5);
                col_max_1[5] = _shfl_94;
                float _shfl_95 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 6);
                col_max_1[6] = _shfl_95;
                float _shfl_96 = __shfl_sync(0xFFFFFFFF, m_scaled_1, 7);
                col_max_1[7] = _shfl_96;
                float _exp2_17 = approx_exp2(score_values_1[0] * softmax_scale_log2_1 - col_max_1[0]);
                score_values_1[0] = _exp2_17;
                float _exp2_18 = approx_exp2(score_values_1[1] * softmax_scale_log2_1 - col_max_1[1]);
                score_values_1[1] = _exp2_18;
                float _exp2_19 = approx_exp2(score_values_1[2] * softmax_scale_log2_1 - col_max_1[2]);
                score_values_1[2] = _exp2_19;
                float _exp2_20 = approx_exp2(score_values_1[3] * softmax_scale_log2_1 - col_max_1[3]);
                score_values_1[3] = _exp2_20;
                float _exp2_21 = approx_exp2(score_values_1[4] * softmax_scale_log2_1 - col_max_1[4]);
                score_values_1[4] = _exp2_21;
                float _exp2_22 = approx_exp2(score_values_1[5] * softmax_scale_log2_1 - col_max_1[5]);
                score_values_1[5] = _exp2_22;
                float _exp2_23 = approx_exp2(score_values_1[6] * softmax_scale_log2_1 - col_max_1[6]);
                score_values_1[6] = _exp2_23;
                float _exp2_24 = approx_exp2(score_values_1[7] * softmax_scale_log2_1 - col_max_1[7]);
                score_values_1[7] = _exp2_24;
                {
                    uint16_t _fp8_pair_14;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_14) : "f"(0.0f), "f"(score_values_1[0]));
                    uint32_t _byte_14 = (uint32_t)(_fp8_pair_14 & 0xFF);
                    uint32_t _addr_14 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2048 + row_1 ^ (2048 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_14), "r"(_byte_14) : "memory");
                }
                float _fp8_rt_16;
                uint16_t _e4m3x2_15;
                uint32_t _f16x2_15;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_15) : "f"(0.0f), "f"(score_values_1[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_15) : "h"(_e4m3x2_15));
                uint16_t _fp8_h0_15 = (uint16_t)(_f16x2_15 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_16) : "h"(_fp8_h0_15));
                tr_vals_1[0] = _fp8_rt_16;
                {
                    uint16_t _fp8_pair_16;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_16) : "f"(0.0f), "f"(score_values_1[1]));
                    uint32_t _byte_16 = (uint32_t)(_fp8_pair_16 & 0xFF);
                    uint32_t _addr_16 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2176 + row_1 ^ (2176 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_16), "r"(_byte_16) : "memory");
                }
                float _fp8_rt_17;
                uint16_t _e4m3x2_17;
                uint32_t _f16x2_17;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_17) : "f"(0.0f), "f"(score_values_1[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_17) : "h"(_e4m3x2_17));
                uint16_t _fp8_h0_17 = (uint16_t)(_f16x2_17 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_17) : "h"(_fp8_h0_17));
                tr_vals_1[1] = _fp8_rt_17;
                {
                    uint16_t _fp8_pair_18;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_18) : "f"(0.0f), "f"(score_values_1[2]));
                    uint32_t _byte_18 = (uint32_t)(_fp8_pair_18 & 0xFF);
                    uint32_t _addr_18 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2304 + row_1 ^ (2304 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_18), "r"(_byte_18) : "memory");
                }
                float _fp8_rt_18;
                uint16_t _e4m3x2_19;
                uint32_t _f16x2_19;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_19) : "f"(0.0f), "f"(score_values_1[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_19) : "h"(_e4m3x2_19));
                uint16_t _fp8_h0_19 = (uint16_t)(_f16x2_19 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_18) : "h"(_fp8_h0_19));
                tr_vals_1[2] = _fp8_rt_18;
                {
                    uint16_t _fp8_pair_20;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_20) : "f"(0.0f), "f"(score_values_1[3]));
                    uint32_t _byte_20 = (uint32_t)(_fp8_pair_20 & 0xFF);
                    uint32_t _addr_20 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2432 + row_1 ^ (2432 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_20), "r"(_byte_20) : "memory");
                }
                float _fp8_rt_19;
                uint16_t _e4m3x2_21;
                uint32_t _f16x2_21;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_21) : "f"(0.0f), "f"(score_values_1[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_21) : "h"(_e4m3x2_21));
                uint16_t _fp8_h0_21 = (uint16_t)(_f16x2_21 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_19) : "h"(_fp8_h0_21));
                tr_vals_1[3] = _fp8_rt_19;
                {
                    uint16_t _fp8_pair_22;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_22) : "f"(0.0f), "f"(score_values_1[4]));
                    uint32_t _byte_22 = (uint32_t)(_fp8_pair_22 & 0xFF);
                    uint32_t _addr_22 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2560 + row_1 ^ (2560 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_22), "r"(_byte_22) : "memory");
                }
                float _fp8_rt_20;
                uint16_t _e4m3x2_23;
                uint32_t _f16x2_23;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_23) : "f"(0.0f), "f"(score_values_1[4]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_23) : "h"(_e4m3x2_23));
                uint16_t _fp8_h0_23 = (uint16_t)(_f16x2_23 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_20) : "h"(_fp8_h0_23));
                tr_vals_1[4] = _fp8_rt_20;
                {
                    uint16_t _fp8_pair_24;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_24) : "f"(0.0f), "f"(score_values_1[5]));
                    uint32_t _byte_24 = (uint32_t)(_fp8_pair_24 & 0xFF);
                    uint32_t _addr_24 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2688 + row_1 ^ (2688 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_24), "r"(_byte_24) : "memory");
                }
                float _fp8_rt_21;
                uint16_t _e4m3x2_25;
                uint32_t _f16x2_25;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_25) : "f"(0.0f), "f"(score_values_1[5]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_25) : "h"(_e4m3x2_25));
                uint16_t _fp8_h0_25 = (uint16_t)(_f16x2_25 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_21) : "h"(_fp8_h0_25));
                tr_vals_1[5] = _fp8_rt_21;
                {
                    uint16_t _fp8_pair_26;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_26) : "f"(0.0f), "f"(score_values_1[6]));
                    uint32_t _byte_26 = (uint32_t)(_fp8_pair_26 & 0xFF);
                    uint32_t _addr_26 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2816 + row_1 ^ (2816 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_26), "r"(_byte_26) : "memory");
                }
                float _fp8_rt_22;
                uint16_t _e4m3x2_27;
                uint32_t _f16x2_27;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_27) : "f"(0.0f), "f"(score_values_1[6]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_27) : "h"(_e4m3x2_27));
                uint16_t _fp8_h0_27 = (uint16_t)(_f16x2_27 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_22) : "h"(_fp8_h0_27));
                tr_vals_1[6] = _fp8_rt_22;
                {
                    uint16_t _fp8_pair_28;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_28) : "f"(0.0f), "f"(score_values_1[7]));
                    uint32_t _byte_28 = (uint32_t)(_fp8_pair_28 & 0xFF);
                    uint32_t _addr_28 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(2944 + row_1 ^ (2944 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_28), "r"(_byte_28) : "memory");
                }
                float _fp8_rt_23;
                uint16_t _e4m3x2_29;
                uint32_t _f16x2_29;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_29) : "f"(0.0f), "f"(score_values_1[7]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_29) : "h"(_e4m3x2_29));
                uint16_t _fp8_h0_29 = (uint16_t)(_f16x2_29 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_23) : "h"(_fp8_h0_29));
                tr_vals_1[7] = _fp8_rt_23;
                int hi_bit_21_1 = lane & 16;
                float send_22_1 = ((hi_bit_21_1 != 0) ? score_values_1[0] : score_values_1[4]);
                float keep_23_1 = ((hi_bit_21_1 != 0) ? score_values_1[4] : score_values_1[0]);
                float _shfl_97 = __shfl_sync(0xFFFFFFFF, send_22_1, lane ^ 16);
                float recv_24_1 = _shfl_97;
                score_values_1[0] = keep_23_1 + recv_24_1;
                float send_25_1 = ((hi_bit_21_1 != 0) ? score_values_1[1] : score_values_1[5]);
                float keep_26_1 = ((hi_bit_21_1 != 0) ? score_values_1[5] : score_values_1[1]);
                float _shfl_98 = __shfl_sync(0xFFFFFFFF, send_25_1, lane ^ 16);
                float recv_27_1 = _shfl_98;
                score_values_1[1] = keep_26_1 + recv_27_1;
                float send_28_1 = ((hi_bit_21_1 != 0) ? score_values_1[2] : score_values_1[6]);
                float keep_29_1 = ((hi_bit_21_1 != 0) ? score_values_1[6] : score_values_1[2]);
                float _shfl_99 = __shfl_sync(0xFFFFFFFF, send_28_1, lane ^ 16);
                float recv_30_1 = _shfl_99;
                score_values_1[2] = keep_29_1 + recv_30_1;
                float send_31_1 = ((hi_bit_21_1 != 0) ? score_values_1[3] : score_values_1[7]);
                float keep_32_1 = ((hi_bit_21_1 != 0) ? score_values_1[7] : score_values_1[3]);
                float _shfl_100 = __shfl_sync(0xFFFFFFFF, send_31_1, lane ^ 16);
                float recv_33_1 = _shfl_100;
                score_values_1[3] = keep_32_1 + recv_33_1;
                int hi_bit_34_1 = lane & 8;
                float send_35_1 = ((hi_bit_34_1 != 0) ? score_values_1[0] : score_values_1[2]);
                float keep_36_1 = ((hi_bit_34_1 != 0) ? score_values_1[2] : score_values_1[0]);
                float _shfl_101 = __shfl_sync(0xFFFFFFFF, send_35_1, lane ^ 8);
                float recv_37_1 = _shfl_101;
                score_values_1[0] = keep_36_1 + recv_37_1;
                float send_38_1 = ((hi_bit_34_1 != 0) ? score_values_1[1] : score_values_1[3]);
                float keep_39_1 = ((hi_bit_34_1 != 0) ? score_values_1[3] : score_values_1[1]);
                float _shfl_102 = __shfl_sync(0xFFFFFFFF, send_38_1, lane ^ 8);
                float recv_40_1 = _shfl_102;
                score_values_1[1] = keep_39_1 + recv_40_1;
                int hi_bit_41_1 = lane & 4;
                float send_42_1 = ((hi_bit_41_1 != 0) ? score_values_1[0] : score_values_1[1]);
                float keep_43_1 = ((hi_bit_41_1 != 0) ? score_values_1[1] : score_values_1[0]);
                float _shfl_103 = __shfl_sync(0xFFFFFFFF, send_42_1, lane ^ 4);
                float recv_44_1 = _shfl_103;
                score_values_1[0] = keep_43_1 + recv_44_1;
                float _shfl_104 = __shfl_sync(0xFFFFFFFF, score_values_1[0], lane ^ 2);
                float other_45 = _shfl_104;
                score_values_1[0] = score_values_1[0] + other_45;
                float _shfl_105 = __shfl_sync(0xFFFFFFFF, score_values_1[0], lane ^ 1);
                float other_46 = _shfl_105;
                score_values_1[0] = score_values_1[0] + other_46;
                int hi_bit_47 = lane & 16;
                float send_48 = ((hi_bit_47 != 0) ? tr_vals_1[0] : tr_vals_1[4]);
                float keep_49 = ((hi_bit_47 != 0) ? tr_vals_1[4] : tr_vals_1[0]);
                float _shfl_106 = __shfl_sync(0xFFFFFFFF, send_48, lane ^ 16);
                float recv_50 = _shfl_106;
                tr_vals_1[0] = keep_49 + recv_50;
                float send_51 = ((hi_bit_47 != 0) ? tr_vals_1[1] : tr_vals_1[5]);
                float keep_52 = ((hi_bit_47 != 0) ? tr_vals_1[5] : tr_vals_1[1]);
                float _shfl_107 = __shfl_sync(0xFFFFFFFF, send_51, lane ^ 16);
                float recv_53 = _shfl_107;
                tr_vals_1[1] = keep_52 + recv_53;
                float send_54 = ((hi_bit_47 != 0) ? tr_vals_1[2] : tr_vals_1[6]);
                float keep_55 = ((hi_bit_47 != 0) ? tr_vals_1[6] : tr_vals_1[2]);
                float _shfl_108 = __shfl_sync(0xFFFFFFFF, send_54, lane ^ 16);
                float recv_56 = _shfl_108;
                tr_vals_1[2] = keep_55 + recv_56;
                float send_57 = ((hi_bit_47 != 0) ? tr_vals_1[3] : tr_vals_1[7]);
                float keep_58 = ((hi_bit_47 != 0) ? tr_vals_1[7] : tr_vals_1[3]);
                float _shfl_109 = __shfl_sync(0xFFFFFFFF, send_57, lane ^ 16);
                float recv_59 = _shfl_109;
                tr_vals_1[3] = keep_58 + recv_59;
                int hi_bit_60 = lane & 8;
                float send_61_1 = ((hi_bit_60 != 0) ? tr_vals_1[0] : tr_vals_1[2]);
                float keep_62_1 = ((hi_bit_60 != 0) ? tr_vals_1[2] : tr_vals_1[0]);
                float _shfl_110 = __shfl_sync(0xFFFFFFFF, send_61_1, lane ^ 8);
                float recv_63_1 = _shfl_110;
                tr_vals_1[0] = keep_62_1 + recv_63_1;
                float send_64_1 = ((hi_bit_60 != 0) ? tr_vals_1[1] : tr_vals_1[3]);
                float keep_65_1 = ((hi_bit_60 != 0) ? tr_vals_1[3] : tr_vals_1[1]);
                float _shfl_111 = __shfl_sync(0xFFFFFFFF, send_64_1, lane ^ 8);
                float recv_66_1 = _shfl_111;
                tr_vals_1[1] = keep_65_1 + recv_66_1;
                int hi_bit_67 = lane & 4;
                float send_68 = ((hi_bit_67 != 0) ? tr_vals_1[0] : tr_vals_1[1]);
                float keep_69 = ((hi_bit_67 != 0) ? tr_vals_1[1] : tr_vals_1[0]);
                float _shfl_112 = __shfl_sync(0xFFFFFFFF, send_68, lane ^ 4);
                float recv_70 = _shfl_112;
                tr_vals_1[0] = keep_69 + recv_70;
                float _shfl_113 = __shfl_sync(0xFFFFFFFF, tr_vals_1[0], lane ^ 2);
                float other_71 = _shfl_113;
                tr_vals_1[0] = tr_vals_1[0] + other_71;
                float _shfl_114 = __shfl_sync(0xFFFFFFFF, tr_vals_1[0], lane ^ 1);
                float other_72 = _shfl_114;
                tr_vals_1[0] = tr_vals_1[0] + other_72;
                if ((lane & 3) == 0) {
                    smem_psum[(4 + local_warp_1) * 16 + (lane >> 2)] = score_values_1[0];
                    smem_rsum[(4 + local_warp_1) * 16 + (lane >> 2)] = tr_vals_1[0];
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                asm volatile("barrier.sync 11, 128;" ::: "memory");
                float norm_lane_1 = 0.0f;
                if (lane < 8) {
                    float _exp2_25 = approx_exp2(sink_lane_1 - m_scaled_1);
                    float sink_term_1 = _exp2_25;
                    float col_sum_1 = smem_psum[64 + lane] + smem_psum[80 + lane] + smem_psum[96 + lane] + smem_psum[112 + lane] + sink_term_1;
                    float denom_1 = smem_rsum[64 + lane] + smem_rsum[80 + lane] + smem_rsum[96 + lane] + smem_rsum[112 + lane] + sink_term_1;
                    if (denom_1 > 0.0f) {
                        float _rcp_1 = approx_rcp(denom_1);
                        norm_lane_1 = _rcp_1 * output_scale_1;
                    }
                    if (local_warp_1 == 0) {
                        if (o_chunk_1 == 0 && head_base_1 + 16 + lane < num_heads) {
                            int lse_offset_1 = (query_idx_1 * num_heads + head_base_1 + 16 + lane) * num_splits + split_idx_1;
                            float _log2_1;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(col_sum_1));
                            partial_lse[lse_offset_1] = ((col_sum_1 > 0.0f) ? (m_scaled_1 + _log2_1) * lse_partial_scale : -CAKE_INF);
                        }
                    }
                }
                float norm_c_1[8];
                float _shfl_115 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 0);
                norm_c_1[0] = _shfl_115;
                float _shfl_116 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 1);
                norm_c_1[1] = _shfl_116;
                float _shfl_117 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 2);
                norm_c_1[2] = _shfl_117;
                float _shfl_118 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 3);
                norm_c_1[3] = _shfl_118;
                float _shfl_119 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 4);
                norm_c_1[4] = _shfl_119;
                float _shfl_120 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 5);
                norm_c_1[5] = _shfl_120;
                float _shfl_121 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 6);
                norm_c_1[6] = _shfl_121;
                float _shfl_122 = __shfl_sync(0xFFFFFFFF, norm_lane_1, 7);
                norm_c_1[7] = _shfl_122;
                float o_values_1[8];
                int dim_1 = 0;
                long long out_off_1 = 0;
                mbarrier_wait_hint(o_full_addr, _phase_o_full_0_1, 10000000);
                _phase_o_full_0_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x8(&o_values_1[0], taddr + 32 + 16 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_1 = o_chunk_1 * 128 + row_1;
                if (head_base_1 + 16 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[0] * norm_c_1[0];
                }
                if (head_base_1 + 16 + 1 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 1) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[1] * norm_c_1[1];
                }
                if (head_base_1 + 16 + 2 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 2) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[2] * norm_c_1[2];
                }
                if (head_base_1 + 16 + 3 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 3) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[3] * norm_c_1[3];
                }
                if (head_base_1 + 16 + 4 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 4) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[4] * norm_c_1[4];
                }
                if (head_base_1 + 16 + 5 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 5) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[5] * norm_c_1[5];
                }
                if (head_base_1 + 16 + 6 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 6) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[6] * norm_c_1[6];
                }
                if (head_base_1 + 16 + 7 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 16 + 7) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[7] * norm_c_1[7];
                }
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }
    // ---- Role: compute2 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // compute2_main
            const int local_warp_2 = warp - 8;
            int o_chunk_2 = blockIdx.x % 4;
            int work_idx_2 = blockIdx.x / 4;
            int head_tile_2 = work_idx_2 % num_head_tiles;
            int split_work_2 = work_idx_2 / num_head_tiles;
            int split_idx_2 = split_work_2 % num_splits;
            int query_idx_2 = split_work_2 / num_splits;
            int head_base_2 = head_tile_2 * 128;
            const int row_2 = local_warp_2 * 32 + lane;
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
            int strip_2 = smem_sfs_addr + (unsigned int)(row_2 * 32);
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            const int gather_tid_2 = warp * 32 + lane;
            const int g_chunk_2 = gather_tid_2 % 24;
            const int g_row0_2 = gather_tid_2 / 24;
            const int g_kind_2 = ((g_chunk_2 < 14) ? 0 : ((g_chunk_2 < 22) ? 1 : 2));
            long long g_src_off_2 = (long long)(((g_kind_2 < 2) ? 16 * g_chunk_2 : 16 * (g_chunk_2 - 14 - 8)));
            int g_dst_k_2 = smem_kf4_addr + (unsigned int)(g_chunk_2 / 8 * 16384) + (unsigned int)(g_row0_2 * 128 + (g_chunk_2 % 8 * 16 ^ g_row0_2 % 8 * 16));
            int g_dst_r_2 = smem_krope_addr + (unsigned int)(g_row0_2 * 128 + ((g_chunk_2 - 14) * 16 ^ g_row0_2 % 8 * 16));
            int g_dst_s_2 = smem_sfs_addr + (unsigned int)(g_row0_2 * 32) + (unsigned int)(16 * (g_chunk_2 - 14 - 8));
            int g_dst0_2 = ((g_kind_2 == 0) ? g_dst_k_2 : ((g_kind_2 == 1) ? g_dst_r_2 : g_dst_s_2));
            const int g_row_bytes_2 = ((g_kind_2 < 2) ? 128 : 32);
            unsigned int w4_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)(g_row0_2 * 16)));
            long long od_3 = (long long)w4_2[1] << 32 | (long long)w4_2[0];
            long long osf_3 = (long long)w4_2[3] << 32 | (long long)w4_2[2];
            long long goff_2 = ((g_kind_2 < 2) ? od_3 : osf_3);
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
            long long goff_3_2 = ((g_kind_2 < 2) ? od_1_2 : osf_2_2);
            if (goff_3_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 16 * g_row_bytes_2), "l"(cache_2 + (goff_3_2 + g_src_off_2)));
            }
            unsigned int w4_4_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 32) * 16)));
            long long od_5_2 = (long long)w4_4_2[1] << 32 | (long long)w4_4_2[0];
            long long osf_6_2 = (long long)w4_4_2[3] << 32 | (long long)w4_4_2[2];
            long long goff_7_2 = ((g_kind_2 < 2) ? od_5_2 : osf_6_2);
            if (goff_7_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 32 * g_row_bytes_2), "l"(cache_2 + (goff_7_2 + g_src_off_2)));
            }
            unsigned int w4_8_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_8_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_8_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 48) * 16)));
            long long od_9_2 = (long long)w4_8_2[1] << 32 | (long long)w4_8_2[0];
            long long osf_10_2 = (long long)w4_8_2[3] << 32 | (long long)w4_8_2[2];
            long long goff_11_2 = ((g_kind_2 < 2) ? od_9_2 : osf_10_2);
            if (goff_11_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 48 * g_row_bytes_2), "l"(cache_2 + (goff_11_2 + g_src_off_2)));
            }
            unsigned int w4_12_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_12_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 64) * 16)));
            long long od_13_2 = (long long)w4_12_2[1] << 32 | (long long)w4_12_2[0];
            long long osf_14_2 = (long long)w4_12_2[3] << 32 | (long long)w4_12_2[2];
            long long goff_15_2 = ((g_kind_2 < 2) ? od_13_2 : osf_14_2);
            if (goff_15_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 64 * g_row_bytes_2), "l"(cache_2 + (goff_15_2 + g_src_off_2)));
            }
            unsigned int w4_16_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_16_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_16_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 80) * 16)));
            long long od_17_2 = (long long)w4_16_2[1] << 32 | (long long)w4_16_2[0];
            long long osf_18_2 = (long long)w4_16_2[3] << 32 | (long long)w4_16_2[2];
            long long goff_19_2 = ((g_kind_2 < 2) ? od_17_2 : osf_18_2);
            if (goff_19_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 80 * g_row_bytes_2), "l"(cache_2 + (goff_19_2 + g_src_off_2)));
            }
            unsigned int w4_20_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_20_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_20_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 96) * 16)));
            long long od_21_2 = (long long)w4_20_2[1] << 32 | (long long)w4_20_2[0];
            long long osf_22_2 = (long long)w4_20_2[3] << 32 | (long long)w4_20_2[2];
            long long goff_23_2 = ((g_kind_2 < 2) ? od_21_2 : osf_22_2);
            if (goff_23_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 96 * g_row_bytes_2), "l"(cache_2 + (goff_23_2 + g_src_off_2)));
            }
            unsigned int w4_24_2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&w4_24_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_24_2[(0) + 3]))
                : "r"(smem_rowoff_addr + (unsigned int)((g_row0_2 + 112) * 16)));
            long long od_25_2 = (long long)w4_24_2[1] << 32 | (long long)w4_24_2[0];
            long long osf_26_2 = (long long)w4_24_2[3] << 32 | (long long)w4_24_2[2];
            long long goff_27_2 = ((g_kind_2 < 2) ? od_25_2 : osf_26_2);
            if (goff_27_2 >= 0) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(g_dst0_2 + 112 * g_row_bytes_2), "l"(cache_2 + (goff_27_2 + g_src_off_2)));
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
                    if (head_base_2 + q_block_2 * 32 < num_heads && q_row_2 < 32) {
                        int q_row_addr_2 = smem_qstage_addr + (unsigned int)(kset_2 * 4096) + (unsigned int)(q_row_2 * 128);
                        unsigned int sf_word_2 = 0;
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
                            float _max_93 = max_noftz(_fabs_64, _fabs_65);
                            m8_2[0] = _max_93;
                            float _fabs_66 = fabsf(qv_3[2]);
                            float _fabs_67 = fabsf(qv_3[3]);
                            float _max_94 = max_noftz(_fabs_66, _fabs_67);
                            m8_2[1] = _max_94;
                            float _fabs_68 = fabsf(qv_3[4]);
                            float _fabs_69 = fabsf(qv_3[5]);
                            float _max_95 = max_noftz(_fabs_68, _fabs_69);
                            m8_2[2] = _max_95;
                            float _fabs_70 = fabsf(qv_3[6]);
                            float _fabs_71 = fabsf(qv_3[7]);
                            float _max_96 = max_noftz(_fabs_70, _fabs_71);
                            m8_2[3] = _max_96;
                            float _fabs_72 = fabsf(qv_3[8]);
                            float _fabs_73 = fabsf(qv_3[9]);
                            float _max_97 = max_noftz(_fabs_72, _fabs_73);
                            m8_2[4] = _max_97;
                            float _fabs_74 = fabsf(qv_3[10]);
                            float _fabs_75 = fabsf(qv_3[11]);
                            float _max_98 = max_noftz(_fabs_74, _fabs_75);
                            m8_2[5] = _max_98;
                            float _fabs_76 = fabsf(qv_3[12]);
                            float _fabs_77 = fabsf(qv_3[13]);
                            float _max_99 = max_noftz(_fabs_76, _fabs_77);
                            m8_2[6] = _max_99;
                            float _fabs_78 = fabsf(qv_3[14]);
                            float _fabs_79 = fabsf(qv_3[15]);
                            float _max_100 = max_noftz(_fabs_78, _fabs_79);
                            m8_2[7] = _max_100;
                            float m4_2[4];
                            float _max_101 = max_noftz(m8_2[0], m8_2[1]);
                            m4_2[0] = _max_101;
                            float _max_102 = max_noftz(m8_2[2], m8_2[3]);
                            m4_2[1] = _max_102;
                            float _max_103 = max_noftz(m8_2[4], m8_2[5]);
                            m4_2[2] = _max_103;
                            float _max_104 = max_noftz(m8_2[6], m8_2[7]);
                            m4_2[3] = _max_104;
                            float _max_105 = max_noftz(m4_2[0], m4_2[1]);
                            float _max_106 = max_noftz(m4_2[2], m4_2[3]);
                            float _max_107 = max_noftz(_max_105, _max_106);
                            float amax_2 = _max_107;
                            float sc_2 = amax_2 * inv_six_2;
                            uint16_t _e4m3x2_f32_100;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_100) : "f"(0.0f), "f"(sc_2));
                            uint16_t sc_pair_2 = _e4m3x2_f32_100;
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
                            sf_word_2 = sf_word_2 | sc_byte_2 << (unsigned int)(8 * (2 * bp_2));
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
                            float _max_108 = max_noftz(_fabs_80, _fabs_81);
                            m8_3_2[0] = _max_108;
                            float _fabs_82 = fabsf(qv_2_2[2]);
                            float _fabs_83 = fabsf(qv_2_2[3]);
                            float _max_109 = max_noftz(_fabs_82, _fabs_83);
                            m8_3_2[1] = _max_109;
                            float _fabs_84 = fabsf(qv_2_2[4]);
                            float _fabs_85 = fabsf(qv_2_2[5]);
                            float _max_110 = max_noftz(_fabs_84, _fabs_85);
                            m8_3_2[2] = _max_110;
                            float _fabs_86 = fabsf(qv_2_2[6]);
                            float _fabs_87 = fabsf(qv_2_2[7]);
                            float _max_111 = max_noftz(_fabs_86, _fabs_87);
                            m8_3_2[3] = _max_111;
                            float _fabs_88 = fabsf(qv_2_2[8]);
                            float _fabs_89 = fabsf(qv_2_2[9]);
                            float _max_112 = max_noftz(_fabs_88, _fabs_89);
                            m8_3_2[4] = _max_112;
                            float _fabs_90 = fabsf(qv_2_2[10]);
                            float _fabs_91 = fabsf(qv_2_2[11]);
                            float _max_113 = max_noftz(_fabs_90, _fabs_91);
                            m8_3_2[5] = _max_113;
                            float _fabs_92 = fabsf(qv_2_2[12]);
                            float _fabs_93 = fabsf(qv_2_2[13]);
                            float _max_114 = max_noftz(_fabs_92, _fabs_93);
                            m8_3_2[6] = _max_114;
                            float _fabs_94 = fabsf(qv_2_2[14]);
                            float _fabs_95 = fabsf(qv_2_2[15]);
                            float _max_115 = max_noftz(_fabs_94, _fabs_95);
                            m8_3_2[7] = _max_115;
                            float m4_4_2[4];
                            float _max_116 = max_noftz(m8_3_2[0], m8_3_2[1]);
                            m4_4_2[0] = _max_116;
                            float _max_117 = max_noftz(m8_3_2[2], m8_3_2[3]);
                            m4_4_2[1] = _max_117;
                            float _max_118 = max_noftz(m8_3_2[4], m8_3_2[5]);
                            m4_4_2[2] = _max_118;
                            float _max_119 = max_noftz(m8_3_2[6], m8_3_2[7]);
                            m4_4_2[3] = _max_119;
                            float _max_120 = max_noftz(m4_4_2[0], m4_4_2[1]);
                            float _max_121 = max_noftz(m4_4_2[2], m4_4_2[3]);
                            float _max_122 = max_noftz(_max_120, _max_121);
                            float amax_5_2 = _max_122;
                            float sc_6_2 = amax_5_2 * inv_six_2;
                            uint16_t _e4m3x2_f32_101;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_101) : "f"(0.0f), "f"(sc_6_2));
                            uint16_t sc_pair_7_2 = _e4m3x2_f32_101;
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
                            sf_word_2 = sf_word_2 | sc_byte_8_2 << (unsigned int)(8 * (2 * bp_2 + 1));
                            int chunk_2 = 2 * kset_2 + bp_2;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                "r"(smem_qf4_addr + (unsigned int)(chunk_2 / 8 * 16384 + (q_row_2 * 128 + (chunk_2 % 8 * 16 ^ q_row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&words_2[0])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 3])));
                        }
                        smem_qsf32[kset_2 / 4 * 2048 + q_row_2 % 32 / 8 * 512 + kset_2 % 4 * 128 + q_row_2 % 8 * 16 + q_row_2 / 32 % 4 * 4 >> 2] = sf_word_2;
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
            asm volatile("cp.async.wait_group 0;");
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            if (valid_2 != 0) {
                int vblock_3 = 8 * o_chunk_2 + 6;
                unsigned int v8_4[4];
                if (vblock_3 < 28) {
                    int vchunk_6 = vblock_3 >> 1;
                    int vhalf_6 = vblock_3 & 1;
                    unsigned int kraw_6[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_6[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_6 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_6 % 8 * 16 ^ row_2 % 8 * 16)) + (unsigned int)(8 * vhalf_6)));
                    unsigned int sfw32_6 = smem_sfs32[row_2 * 32 + vblock_3 >> 2];
                    unsigned int scale_6 = sfw32_6 >> (unsigned int)(8 * (vblock_3 & 3)) & 255;
                    {
                        v8_4[0] = cake_dsv4_qmul4_portable<5>(kraw_6[0], scale_6);
                    }
                    {
                        v8_4[1] = cake_dsv4_qmul4_portable<6>(kraw_6[0], scale_6);
                    }
                    {
                        v8_4[2] = cake_dsv4_qmul4_portable<5>(kraw_6[1], scale_6);
                    }
                    {
                        v8_4[3] = cake_dsv4_qmul4_portable<6>(kraw_6[1], scale_6);
                    }
                } else {
                    int rblock_2 = vblock_3 - 28;
                    unsigned int rope_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + (2 * rblock_2 * 16 ^ row_2 % 8 * 16))));
                    float lo_2 = __uint_as_float(rope_2[0] << 16);
                    float hi_3 = __uint_as_float(rope_2[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_110;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_110) : "f"(hi_3), "f"(lo_2));
                    uint16_t pair_3 = _e4m3x2_f32_110;
                    {
                        v8_4[0] = (unsigned int)pair_3;
                    }
                    float lo_0_2 = __uint_as_float(rope_2[1] << 16);
                    float hi_1_2 = __uint_as_float(rope_2[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_111;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_111) : "f"(hi_1_2), "f"(lo_0_2));
                    uint16_t pair_2_2 = _e4m3x2_f32_111;
                    {
                        v8_4[0] = v8_4[0] | (unsigned int)pair_2_2 << 16;
                    }
                    float lo_3_2 = __uint_as_float(rope_2[2] << 16);
                    float hi_4_2 = __uint_as_float(rope_2[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_112;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_112) : "f"(hi_4_2), "f"(lo_3_2));
                    uint16_t pair_5_2 = _e4m3x2_f32_112;
                    {
                        v8_4[1] = (unsigned int)pair_5_2;
                    }
                    float lo_6_2 = __uint_as_float(rope_2[3] << 16);
                    float hi_7_2 = __uint_as_float(rope_2[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_113;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_113) : "f"(hi_7_2), "f"(lo_6_2));
                    uint16_t pair_8_2 = _e4m3x2_f32_113;
                    {
                        v8_4[1] = v8_4[1] | (unsigned int)pair_8_2 << 16;
                    }
                    unsigned int rope_9_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_2 + 1) * 16 ^ row_2 % 8 * 16))));
                    float lo_10_2 = __uint_as_float(rope_9_2[0] << 16);
                    float hi_11_2 = __uint_as_float(rope_9_2[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_114;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_114) : "f"(hi_11_2), "f"(lo_10_2));
                    uint16_t pair_12_2 = _e4m3x2_f32_114;
                    {
                        v8_4[2] = (unsigned int)pair_12_2;
                    }
                    float lo_13_2 = __uint_as_float(rope_9_2[1] << 16);
                    float hi_14_2 = __uint_as_float(rope_9_2[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_115;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_115) : "f"(hi_14_2), "f"(lo_13_2));
                    uint16_t pair_15_2 = _e4m3x2_f32_115;
                    {
                        v8_4[2] = v8_4[2] | (unsigned int)pair_15_2 << 16;
                    }
                    float lo_16_2 = __uint_as_float(rope_9_2[2] << 16);
                    float hi_17_2 = __uint_as_float(rope_9_2[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_116;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_116) : "f"(hi_17_2), "f"(lo_16_2));
                    uint16_t pair_18_2 = _e4m3x2_f32_116;
                    {
                        v8_4[3] = (unsigned int)pair_18_2;
                    }
                    float lo_19_2 = __uint_as_float(rope_9_2[3] << 16);
                    float hi_20_2 = __uint_as_float(rope_9_2[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_117;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_117) : "f"(hi_20_2), "f"(lo_19_2));
                    uint16_t pair_21_2 = _e4m3x2_f32_117;
                    {
                        v8_4[3] = v8_4[3] | (unsigned int)pair_21_2 << 16;
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row_2 * 128 + (96 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 3])));
                int vblock_0_2 = 8 * o_chunk_2 + 7;
                unsigned int v8_1_2[4];
                if (vblock_0_2 < 28) {
                    int vchunk_7 = vblock_0_2 >> 1;
                    int vhalf_7 = vblock_0_2 & 1;
                    unsigned int kraw_7[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_7[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_7 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_7 % 8 * 16 ^ row_2 % 8 * 16)) + (unsigned int)(8 * vhalf_7)));
                    unsigned int sfw32_7 = smem_sfs32[row_2 * 32 + vblock_0_2 >> 2];
                    unsigned int scale_7 = sfw32_7 >> (unsigned int)(8 * (vblock_0_2 & 3)) & 255;
                    {
                        v8_1_2[0] = cake_dsv4_qmul4_portable<5>(kraw_7[0], scale_7);
                    }
                    {
                        v8_1_2[1] = cake_dsv4_qmul4_portable<6>(kraw_7[0], scale_7);
                    }
                    {
                        v8_1_2[2] = cake_dsv4_qmul4_portable<5>(kraw_7[1], scale_7);
                    }
                    {
                        v8_1_2[3] = cake_dsv4_qmul4_portable<6>(kraw_7[1], scale_7);
                    }
                } else {
                    int rblock_3 = vblock_0_2 - 28;
                    unsigned int rope_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + (2 * rblock_3 * 16 ^ row_2 % 8 * 16))));
                    float lo_4 = __uint_as_float(rope_3[0] << 16);
                    float hi_5 = __uint_as_float(rope_3[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_126;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_126) : "f"(hi_5), "f"(lo_4));
                    uint16_t pair_4 = _e4m3x2_f32_126;
                    {
                        v8_1_2[0] = (unsigned int)pair_4;
                    }
                    float lo_0_3 = __uint_as_float(rope_3[1] << 16);
                    float hi_1_3 = __uint_as_float(rope_3[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_127;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_127) : "f"(hi_1_3), "f"(lo_0_3));
                    uint16_t pair_2_3 = _e4m3x2_f32_127;
                    {
                        v8_1_2[0] = v8_1_2[0] | (unsigned int)pair_2_3 << 16;
                    }
                    float lo_3_3 = __uint_as_float(rope_3[2] << 16);
                    float hi_4_3 = __uint_as_float(rope_3[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_128;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_128) : "f"(hi_4_3), "f"(lo_3_3));
                    uint16_t pair_5_3 = _e4m3x2_f32_128;
                    {
                        v8_1_2[1] = (unsigned int)pair_5_3;
                    }
                    float lo_6_3 = __uint_as_float(rope_3[3] << 16);
                    float hi_7_3 = __uint_as_float(rope_3[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_129;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_129) : "f"(hi_7_3), "f"(lo_6_3));
                    uint16_t pair_8_3 = _e4m3x2_f32_129;
                    {
                        v8_1_2[1] = v8_1_2[1] | (unsigned int)pair_8_3 << 16;
                    }
                    unsigned int rope_9_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_3 + 1) * 16 ^ row_2 % 8 * 16))));
                    float lo_10_3 = __uint_as_float(rope_9_3[0] << 16);
                    float hi_11_3 = __uint_as_float(rope_9_3[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_130;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_130) : "f"(hi_11_3), "f"(lo_10_3));
                    uint16_t pair_12_3 = _e4m3x2_f32_130;
                    {
                        v8_1_2[2] = (unsigned int)pair_12_3;
                    }
                    float lo_13_3 = __uint_as_float(rope_9_3[1] << 16);
                    float hi_14_3 = __uint_as_float(rope_9_3[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_131;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_131) : "f"(hi_14_3), "f"(lo_13_3));
                    uint16_t pair_15_3 = _e4m3x2_f32_131;
                    {
                        v8_1_2[2] = v8_1_2[2] | (unsigned int)pair_15_3 << 16;
                    }
                    float lo_16_3 = __uint_as_float(rope_9_3[2] << 16);
                    float hi_17_3 = __uint_as_float(rope_9_3[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_132;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_132) : "f"(hi_17_3), "f"(lo_16_3));
                    uint16_t pair_18_3 = _e4m3x2_f32_132;
                    {
                        v8_1_2[3] = (unsigned int)pair_18_3;
                    }
                    float lo_19_3 = __uint_as_float(rope_9_3[3] << 16);
                    float hi_20_3 = __uint_as_float(rope_9_3[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_133;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_133) : "f"(hi_20_3), "f"(lo_19_3));
                    uint16_t pair_21_3 = _e4m3x2_f32_133;
                    {
                        v8_1_2[3] = v8_1_2[3] | (unsigned int)pair_21_3 << 16;
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row_2 * 128 + (112 ^ row_2 % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 3])));
            } else {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row_2 * 128 + (96 ^ row_2 % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row_2 * 128 + (112 ^ row_2 % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            float softmax_scale_log2_2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_2 = bmm2_scale[0];
            unsigned int _phase_s_full_0_2 = 0;
            unsigned int _phase_o_full_0_2 = 0;
            {
                mbarrier_wait_hint(s_full_addr, _phase_s_full_0_2, 10000000);
                _phase_s_full_0_2 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float score_values_2[8];
                tmem_ld_x8(&score_values_2[0], taddr + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (valid_2 == 0) {
                    score_values_2[0] = -CAKE_INF;
                    score_values_2[1] = -CAKE_INF;
                    score_values_2[2] = -CAKE_INF;
                    score_values_2[3] = -CAKE_INF;
                    score_values_2[4] = -CAKE_INF;
                    score_values_2[5] = -CAKE_INF;
                    score_values_2[6] = -CAKE_INF;
                    score_values_2[7] = -CAKE_INF;
                }
                float tr_vals_2[8];
                tr_vals_2[0] = score_values_2[0];
                tr_vals_2[1] = score_values_2[1];
                tr_vals_2[2] = score_values_2[2];
                tr_vals_2[3] = score_values_2[3];
                tr_vals_2[4] = score_values_2[4];
                tr_vals_2[5] = score_values_2[5];
                tr_vals_2[6] = score_values_2[6];
                tr_vals_2[7] = score_values_2[7];
                int hi_bit_2 = lane & 16;
                float send_2 = ((hi_bit_2 != 0) ? tr_vals_2[0] : tr_vals_2[4]);
                float keep_3 = ((hi_bit_2 != 0) ? tr_vals_2[4] : tr_vals_2[0]);
                float _shfl_123 = __shfl_sync(0xFFFFFFFF, send_2, lane ^ 16);
                float recv_3 = _shfl_123;
                float _max_123 = max_noftz(keep_3, recv_3);
                tr_vals_2[0] = _max_123;
                float send_0_2 = ((hi_bit_2 != 0) ? tr_vals_2[1] : tr_vals_2[5]);
                float keep_1_2 = ((hi_bit_2 != 0) ? tr_vals_2[5] : tr_vals_2[1]);
                float _shfl_124 = __shfl_sync(0xFFFFFFFF, send_0_2, lane ^ 16);
                float recv_2_2 = _shfl_124;
                float _max_124 = max_noftz(keep_1_2, recv_2_2);
                tr_vals_2[1] = _max_124;
                float send_3_2 = ((hi_bit_2 != 0) ? tr_vals_2[2] : tr_vals_2[6]);
                float keep_4_2 = ((hi_bit_2 != 0) ? tr_vals_2[6] : tr_vals_2[2]);
                float _shfl_125 = __shfl_sync(0xFFFFFFFF, send_3_2, lane ^ 16);
                float recv_5_2 = _shfl_125;
                float _max_125 = max_noftz(keep_4_2, recv_5_2);
                tr_vals_2[2] = _max_125;
                float send_6_2 = ((hi_bit_2 != 0) ? tr_vals_2[3] : tr_vals_2[7]);
                float keep_7_2 = ((hi_bit_2 != 0) ? tr_vals_2[7] : tr_vals_2[3]);
                float _shfl_126 = __shfl_sync(0xFFFFFFFF, send_6_2, lane ^ 16);
                float recv_8_2 = _shfl_126;
                float _max_126 = max_noftz(keep_7_2, recv_8_2);
                tr_vals_2[3] = _max_126;
                int hi_bit_9_1 = lane & 8;
                float send_10_1 = ((hi_bit_9_1 != 0) ? tr_vals_2[0] : tr_vals_2[2]);
                float keep_11_1 = ((hi_bit_9_1 != 0) ? tr_vals_2[2] : tr_vals_2[0]);
                float _shfl_127 = __shfl_sync(0xFFFFFFFF, send_10_1, lane ^ 8);
                float recv_12_1 = _shfl_127;
                float _max_127 = max_noftz(keep_11_1, recv_12_1);
                tr_vals_2[0] = _max_127;
                float send_13_1 = ((hi_bit_9_1 != 0) ? tr_vals_2[1] : tr_vals_2[3]);
                float keep_14_1 = ((hi_bit_9_1 != 0) ? tr_vals_2[3] : tr_vals_2[1]);
                float _shfl_128 = __shfl_sync(0xFFFFFFFF, send_13_1, lane ^ 8);
                float recv_15_1 = _shfl_128;
                float _max_128 = max_noftz(keep_14_1, recv_15_1);
                tr_vals_2[1] = _max_128;
                int hi_bit_16_1 = lane & 4;
                float send_17_1 = ((hi_bit_16_1 != 0) ? tr_vals_2[0] : tr_vals_2[1]);
                float keep_18_1 = ((hi_bit_16_1 != 0) ? tr_vals_2[1] : tr_vals_2[0]);
                float _shfl_129 = __shfl_sync(0xFFFFFFFF, send_17_1, lane ^ 4);
                float recv_19_1 = _shfl_129;
                float _max_129 = max_noftz(keep_18_1, recv_19_1);
                tr_vals_2[0] = _max_129;
                float _shfl_130 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 2);
                float other_2 = _shfl_130;
                float _max_130 = max_noftz(tr_vals_2[0], other_2);
                tr_vals_2[0] = _max_130;
                float _shfl_131 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 1);
                float other_20_1 = _shfl_131;
                float _max_131 = max_noftz(tr_vals_2[0], other_20_1);
                tr_vals_2[0] = _max_131;
                if ((lane & 3) == 0) {
                    smem_pmax[(8 + local_warp_2) * 16 + (lane >> 2)] = tr_vals_2[0];
                }
                asm volatile("barrier.sync 12, 128;" ::: "memory");
                float m_lane_2 = -CAKE_INF;
                float sink_lane_2 = -CAKE_INF;
                float m_scaled_2 = 0.0f;
                if (lane < 8) {
                    float _max_132 = max_noftz(smem_pmax[128 + lane], smem_pmax[144 + lane]);
                    float _max_133 = max_noftz(smem_pmax[160 + lane], smem_pmax[176 + lane]);
                    float _max_134 = max_noftz(_max_132, _max_133);
                    m_lane_2 = _max_134;
                    if (has_sinks != 0 && split_idx_2 == 0 && head_base_2 + 24 + lane < num_heads) {
                        sink_lane_2 = sinks[head_base_2 + 24 + lane] * 1.4426950408889634f;
                    }
                    float _max_135 = max_noftz(m_lane_2 * softmax_scale_log2_2, sink_lane_2);
                    m_scaled_2 = _max_135;
                    if (m_scaled_2 == -CAKE_INF) {
                        m_scaled_2 = 0.0f;
                    }
                }
                float col_max_2[8];
                float _shfl_132 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 0);
                col_max_2[0] = _shfl_132;
                float _shfl_133 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 1);
                col_max_2[1] = _shfl_133;
                float _shfl_134 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 2);
                col_max_2[2] = _shfl_134;
                float _shfl_135 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 3);
                col_max_2[3] = _shfl_135;
                float _shfl_136 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 4);
                col_max_2[4] = _shfl_136;
                float _shfl_137 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 5);
                col_max_2[5] = _shfl_137;
                float _shfl_138 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 6);
                col_max_2[6] = _shfl_138;
                float _shfl_139 = __shfl_sync(0xFFFFFFFF, m_scaled_2, 7);
                col_max_2[7] = _shfl_139;
                float _exp2_26 = approx_exp2(score_values_2[0] * softmax_scale_log2_2 - col_max_2[0]);
                score_values_2[0] = _exp2_26;
                float _exp2_27 = approx_exp2(score_values_2[1] * softmax_scale_log2_2 - col_max_2[1]);
                score_values_2[1] = _exp2_27;
                float _exp2_28 = approx_exp2(score_values_2[2] * softmax_scale_log2_2 - col_max_2[2]);
                score_values_2[2] = _exp2_28;
                float _exp2_29 = approx_exp2(score_values_2[3] * softmax_scale_log2_2 - col_max_2[3]);
                score_values_2[3] = _exp2_29;
                float _exp2_30 = approx_exp2(score_values_2[4] * softmax_scale_log2_2 - col_max_2[4]);
                score_values_2[4] = _exp2_30;
                float _exp2_31 = approx_exp2(score_values_2[5] * softmax_scale_log2_2 - col_max_2[5]);
                score_values_2[5] = _exp2_31;
                float _exp2_32 = approx_exp2(score_values_2[6] * softmax_scale_log2_2 - col_max_2[6]);
                score_values_2[6] = _exp2_32;
                float _exp2_33 = approx_exp2(score_values_2[7] * softmax_scale_log2_2 - col_max_2[7]);
                score_values_2[7] = _exp2_33;
                {
                    uint16_t _fp8_pair_10;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_10) : "f"(0.0f), "f"(score_values_2[0]));
                    uint32_t _byte_10 = (uint32_t)(_fp8_pair_10 & 0xFF);
                    uint32_t _addr_10 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3072 + row_2 ^ (3072 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_10), "r"(_byte_10) : "memory");
                }
                float _fp8_rt_24;
                uint16_t _e4m3x2_11;
                uint32_t _f16x2_11;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_11) : "f"(0.0f), "f"(score_values_2[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_11) : "h"(_e4m3x2_11));
                uint16_t _fp8_h0_11 = (uint16_t)(_f16x2_11 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_24) : "h"(_fp8_h0_11));
                tr_vals_2[0] = _fp8_rt_24;
                {
                    uint16_t _fp8_pair_12;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_12) : "f"(0.0f), "f"(score_values_2[1]));
                    uint32_t _byte_12 = (uint32_t)(_fp8_pair_12 & 0xFF);
                    uint32_t _addr_12 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3200 + row_2 ^ (3200 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_12), "r"(_byte_12) : "memory");
                }
                float _fp8_rt_25;
                uint16_t _e4m3x2_13;
                uint32_t _f16x2_13;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_13) : "f"(0.0f), "f"(score_values_2[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_13) : "h"(_e4m3x2_13));
                uint16_t _fp8_h0_13 = (uint16_t)(_f16x2_13 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_25) : "h"(_fp8_h0_13));
                tr_vals_2[1] = _fp8_rt_25;
                {
                    uint16_t _fp8_pair_14;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_14) : "f"(0.0f), "f"(score_values_2[2]));
                    uint32_t _byte_14 = (uint32_t)(_fp8_pair_14 & 0xFF);
                    uint32_t _addr_14 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3328 + row_2 ^ (3328 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_14), "r"(_byte_14) : "memory");
                }
                float _fp8_rt_26;
                uint16_t _e4m3x2_15;
                uint32_t _f16x2_15;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_15) : "f"(0.0f), "f"(score_values_2[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_15) : "h"(_e4m3x2_15));
                uint16_t _fp8_h0_15 = (uint16_t)(_f16x2_15 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_26) : "h"(_fp8_h0_15));
                tr_vals_2[2] = _fp8_rt_26;
                {
                    uint16_t _fp8_pair_16;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_16) : "f"(0.0f), "f"(score_values_2[3]));
                    uint32_t _byte_16 = (uint32_t)(_fp8_pair_16 & 0xFF);
                    uint32_t _addr_16 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3456 + row_2 ^ (3456 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_16), "r"(_byte_16) : "memory");
                }
                float _fp8_rt_27;
                uint16_t _e4m3x2_17;
                uint32_t _f16x2_17;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_17) : "f"(0.0f), "f"(score_values_2[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_17) : "h"(_e4m3x2_17));
                uint16_t _fp8_h0_17 = (uint16_t)(_f16x2_17 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_27) : "h"(_fp8_h0_17));
                tr_vals_2[3] = _fp8_rt_27;
                {
                    uint16_t _fp8_pair_18;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_18) : "f"(0.0f), "f"(score_values_2[4]));
                    uint32_t _byte_18 = (uint32_t)(_fp8_pair_18 & 0xFF);
                    uint32_t _addr_18 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3584 + row_2 ^ (3584 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_18), "r"(_byte_18) : "memory");
                }
                float _fp8_rt_28;
                uint16_t _e4m3x2_19;
                uint32_t _f16x2_19;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_19) : "f"(0.0f), "f"(score_values_2[4]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_19) : "h"(_e4m3x2_19));
                uint16_t _fp8_h0_19 = (uint16_t)(_f16x2_19 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_28) : "h"(_fp8_h0_19));
                tr_vals_2[4] = _fp8_rt_28;
                {
                    uint16_t _fp8_pair_20;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_20) : "f"(0.0f), "f"(score_values_2[5]));
                    uint32_t _byte_20 = (uint32_t)(_fp8_pair_20 & 0xFF);
                    uint32_t _addr_20 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3712 + row_2 ^ (3712 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_20), "r"(_byte_20) : "memory");
                }
                float _fp8_rt_29;
                uint16_t _e4m3x2_21;
                uint32_t _f16x2_21;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_21) : "f"(0.0f), "f"(score_values_2[5]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_21) : "h"(_e4m3x2_21));
                uint16_t _fp8_h0_21 = (uint16_t)(_f16x2_21 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_29) : "h"(_fp8_h0_21));
                tr_vals_2[5] = _fp8_rt_29;
                {
                    uint16_t _fp8_pair_22;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_22) : "f"(0.0f), "f"(score_values_2[6]));
                    uint32_t _byte_22 = (uint32_t)(_fp8_pair_22 & 0xFF);
                    uint32_t _addr_22 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3840 + row_2 ^ (3840 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_22), "r"(_byte_22) : "memory");
                }
                float _fp8_rt_30;
                uint16_t _e4m3x2_23;
                uint32_t _f16x2_23;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_23) : "f"(0.0f), "f"(score_values_2[6]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_23) : "h"(_e4m3x2_23));
                uint16_t _fp8_h0_23 = (uint16_t)(_f16x2_23 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_30) : "h"(_fp8_h0_23));
                tr_vals_2[6] = _fp8_rt_30;
                {
                    uint16_t _fp8_pair_24;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_24) : "f"(0.0f), "f"(score_values_2[7]));
                    uint32_t _byte_24 = (uint32_t)(_fp8_pair_24 & 0xFF);
                    uint32_t _addr_24 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(3968 + row_2 ^ (3968 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_24), "r"(_byte_24) : "memory");
                }
                float _fp8_rt_31;
                uint16_t _e4m3x2_25;
                uint32_t _f16x2_25;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_25) : "f"(0.0f), "f"(score_values_2[7]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_25) : "h"(_e4m3x2_25));
                uint16_t _fp8_h0_25 = (uint16_t)(_f16x2_25 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_31) : "h"(_fp8_h0_25));
                tr_vals_2[7] = _fp8_rt_31;
                int hi_bit_21_2 = lane & 16;
                float send_22_2 = ((hi_bit_21_2 != 0) ? score_values_2[0] : score_values_2[4]);
                float keep_23_2 = ((hi_bit_21_2 != 0) ? score_values_2[4] : score_values_2[0]);
                float _shfl_140 = __shfl_sync(0xFFFFFFFF, send_22_2, lane ^ 16);
                float recv_24_2 = _shfl_140;
                score_values_2[0] = keep_23_2 + recv_24_2;
                float send_25_2 = ((hi_bit_21_2 != 0) ? score_values_2[1] : score_values_2[5]);
                float keep_26_2 = ((hi_bit_21_2 != 0) ? score_values_2[5] : score_values_2[1]);
                float _shfl_141 = __shfl_sync(0xFFFFFFFF, send_25_2, lane ^ 16);
                float recv_27_2 = _shfl_141;
                score_values_2[1] = keep_26_2 + recv_27_2;
                float send_28_2 = ((hi_bit_21_2 != 0) ? score_values_2[2] : score_values_2[6]);
                float keep_29_2 = ((hi_bit_21_2 != 0) ? score_values_2[6] : score_values_2[2]);
                float _shfl_142 = __shfl_sync(0xFFFFFFFF, send_28_2, lane ^ 16);
                float recv_30_2 = _shfl_142;
                score_values_2[2] = keep_29_2 + recv_30_2;
                float send_31_2 = ((hi_bit_21_2 != 0) ? score_values_2[3] : score_values_2[7]);
                float keep_32_2 = ((hi_bit_21_2 != 0) ? score_values_2[7] : score_values_2[3]);
                float _shfl_143 = __shfl_sync(0xFFFFFFFF, send_31_2, lane ^ 16);
                float recv_33_2 = _shfl_143;
                score_values_2[3] = keep_32_2 + recv_33_2;
                int hi_bit_34_2 = lane & 8;
                float send_35_2 = ((hi_bit_34_2 != 0) ? score_values_2[0] : score_values_2[2]);
                float keep_36_2 = ((hi_bit_34_2 != 0) ? score_values_2[2] : score_values_2[0]);
                float _shfl_144 = __shfl_sync(0xFFFFFFFF, send_35_2, lane ^ 8);
                float recv_37_2 = _shfl_144;
                score_values_2[0] = keep_36_2 + recv_37_2;
                float send_38_2 = ((hi_bit_34_2 != 0) ? score_values_2[1] : score_values_2[3]);
                float keep_39_2 = ((hi_bit_34_2 != 0) ? score_values_2[3] : score_values_2[1]);
                float _shfl_145 = __shfl_sync(0xFFFFFFFF, send_38_2, lane ^ 8);
                float recv_40_2 = _shfl_145;
                score_values_2[1] = keep_39_2 + recv_40_2;
                int hi_bit_41_2 = lane & 4;
                float send_42_2 = ((hi_bit_41_2 != 0) ? score_values_2[0] : score_values_2[1]);
                float keep_43_2 = ((hi_bit_41_2 != 0) ? score_values_2[1] : score_values_2[0]);
                float _shfl_146 = __shfl_sync(0xFFFFFFFF, send_42_2, lane ^ 4);
                float recv_44_2 = _shfl_146;
                score_values_2[0] = keep_43_2 + recv_44_2;
                float _shfl_147 = __shfl_sync(0xFFFFFFFF, score_values_2[0], lane ^ 2);
                float other_45_1 = _shfl_147;
                score_values_2[0] = score_values_2[0] + other_45_1;
                float _shfl_148 = __shfl_sync(0xFFFFFFFF, score_values_2[0], lane ^ 1);
                float other_46_1 = _shfl_148;
                score_values_2[0] = score_values_2[0] + other_46_1;
                int hi_bit_47_1 = lane & 16;
                float send_48_1 = ((hi_bit_47_1 != 0) ? tr_vals_2[0] : tr_vals_2[4]);
                float keep_49_1 = ((hi_bit_47_1 != 0) ? tr_vals_2[4] : tr_vals_2[0]);
                float _shfl_149 = __shfl_sync(0xFFFFFFFF, send_48_1, lane ^ 16);
                float recv_50_1 = _shfl_149;
                tr_vals_2[0] = keep_49_1 + recv_50_1;
                float send_51_1 = ((hi_bit_47_1 != 0) ? tr_vals_2[1] : tr_vals_2[5]);
                float keep_52_1 = ((hi_bit_47_1 != 0) ? tr_vals_2[5] : tr_vals_2[1]);
                float _shfl_150 = __shfl_sync(0xFFFFFFFF, send_51_1, lane ^ 16);
                float recv_53_1 = _shfl_150;
                tr_vals_2[1] = keep_52_1 + recv_53_1;
                float send_54_1 = ((hi_bit_47_1 != 0) ? tr_vals_2[2] : tr_vals_2[6]);
                float keep_55_1 = ((hi_bit_47_1 != 0) ? tr_vals_2[6] : tr_vals_2[2]);
                float _shfl_151 = __shfl_sync(0xFFFFFFFF, send_54_1, lane ^ 16);
                float recv_56_1 = _shfl_151;
                tr_vals_2[2] = keep_55_1 + recv_56_1;
                float send_57_1 = ((hi_bit_47_1 != 0) ? tr_vals_2[3] : tr_vals_2[7]);
                float keep_58_1 = ((hi_bit_47_1 != 0) ? tr_vals_2[7] : tr_vals_2[3]);
                float _shfl_152 = __shfl_sync(0xFFFFFFFF, send_57_1, lane ^ 16);
                float recv_59_1 = _shfl_152;
                tr_vals_2[3] = keep_58_1 + recv_59_1;
                int hi_bit_60_1 = lane & 8;
                float send_61_2 = ((hi_bit_60_1 != 0) ? tr_vals_2[0] : tr_vals_2[2]);
                float keep_62_2 = ((hi_bit_60_1 != 0) ? tr_vals_2[2] : tr_vals_2[0]);
                float _shfl_153 = __shfl_sync(0xFFFFFFFF, send_61_2, lane ^ 8);
                float recv_63_2 = _shfl_153;
                tr_vals_2[0] = keep_62_2 + recv_63_2;
                float send_64_2 = ((hi_bit_60_1 != 0) ? tr_vals_2[1] : tr_vals_2[3]);
                float keep_65_2 = ((hi_bit_60_1 != 0) ? tr_vals_2[3] : tr_vals_2[1]);
                float _shfl_154 = __shfl_sync(0xFFFFFFFF, send_64_2, lane ^ 8);
                float recv_66_2 = _shfl_154;
                tr_vals_2[1] = keep_65_2 + recv_66_2;
                int hi_bit_67_1 = lane & 4;
                float send_68_1 = ((hi_bit_67_1 != 0) ? tr_vals_2[0] : tr_vals_2[1]);
                float keep_69_1 = ((hi_bit_67_1 != 0) ? tr_vals_2[1] : tr_vals_2[0]);
                float _shfl_155 = __shfl_sync(0xFFFFFFFF, send_68_1, lane ^ 4);
                float recv_70_1 = _shfl_155;
                tr_vals_2[0] = keep_69_1 + recv_70_1;
                float _shfl_156 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 2);
                float other_71_1 = _shfl_156;
                tr_vals_2[0] = tr_vals_2[0] + other_71_1;
                float _shfl_157 = __shfl_sync(0xFFFFFFFF, tr_vals_2[0], lane ^ 1);
                float other_72_1 = _shfl_157;
                tr_vals_2[0] = tr_vals_2[0] + other_72_1;
                if ((lane & 3) == 0) {
                    smem_psum[(8 + local_warp_2) * 16 + (lane >> 2)] = score_values_2[0];
                    smem_rsum[(8 + local_warp_2) * 16 + (lane >> 2)] = tr_vals_2[0];
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                asm volatile("barrier.sync 12, 128;" ::: "memory");
                float norm_lane_2 = 0.0f;
                if (lane < 8) {
                    float _exp2_34 = approx_exp2(sink_lane_2 - m_scaled_2);
                    float sink_term_2 = _exp2_34;
                    float col_sum_2 = smem_psum[128 + lane] + smem_psum[144 + lane] + smem_psum[160 + lane] + smem_psum[176 + lane] + sink_term_2;
                    float denom_2 = smem_rsum[128 + lane] + smem_rsum[144 + lane] + smem_rsum[160 + lane] + smem_rsum[176 + lane] + sink_term_2;
                    if (denom_2 > 0.0f) {
                        float _rcp_2 = approx_rcp(denom_2);
                        norm_lane_2 = _rcp_2 * output_scale_2;
                    }
                    if (local_warp_2 == 0) {
                        if (o_chunk_2 == 0 && head_base_2 + 24 + lane < num_heads) {
                            int lse_offset_2 = (query_idx_2 * num_heads + head_base_2 + 24 + lane) * num_splits + split_idx_2;
                            float _log2_2;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(col_sum_2));
                            partial_lse[lse_offset_2] = ((col_sum_2 > 0.0f) ? (m_scaled_2 + _log2_2) * lse_partial_scale : -CAKE_INF);
                        }
                    }
                }
                float norm_c_2[8];
                float _shfl_158 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 0);
                norm_c_2[0] = _shfl_158;
                float _shfl_159 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 1);
                norm_c_2[1] = _shfl_159;
                float _shfl_160 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 2);
                norm_c_2[2] = _shfl_160;
                float _shfl_161 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 3);
                norm_c_2[3] = _shfl_161;
                float _shfl_162 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 4);
                norm_c_2[4] = _shfl_162;
                float _shfl_163 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 5);
                norm_c_2[5] = _shfl_163;
                float _shfl_164 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 6);
                norm_c_2[6] = _shfl_164;
                float _shfl_165 = __shfl_sync(0xFFFFFFFF, norm_lane_2, 7);
                norm_c_2[7] = _shfl_165;
                float o_values_2[8];
                int dim_2 = 0;
                long long out_off_2 = 0;
                mbarrier_wait_hint(o_full_addr, _phase_o_full_0_2, 10000000);
                _phase_o_full_0_2 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x8(&o_values_2[0], taddr + 32 + 24 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_2 = o_chunk_2 * 128 + row_2;
                if (head_base_2 + 24 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[0] * norm_c_2[0];
                }
                if (head_base_2 + 24 + 1 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 1) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[1] * norm_c_2[1];
                }
                if (head_base_2 + 24 + 2 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 2) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[2] * norm_c_2[2];
                }
                if (head_base_2 + 24 + 3 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 3) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[3] * norm_c_2[3];
                }
                if (head_base_2 + 24 + 4 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 4) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[4] * norm_c_2[4];
                }
                if (head_base_2 + 24 + 5 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 5) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[5] * norm_c_2[5];
                }
                if (head_base_2 + 24 + 6 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 6) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[6] * norm_c_2[6];
                }
                if (head_base_2 + 24 + 7 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 24 + 7) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[7] * norm_c_2[7];
                }
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            unsigned int _phase_q_ready_0 = 0;
            mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0, 10000000);
            _phase_q_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (elect_sync()) {
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb0, make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4))));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 4), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 8), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb0 + 12), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 24)));
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb1, make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 4), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 8), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb1 + 12), make_sf_cp_desc_lo_sbo512((((smem_qsf_addr) >> 4) + 128 + 24)));
            }
            unsigned int _phase_q_rope_full_0 = 0;
            mbarrier_wait_hint(q_rope_full_addr, _phase_q_rope_full_0, 10000000);
            _phase_q_rope_full_0 ^= 1;
            unsigned int _phase_kv_full_0 = 0;
            mbarrier_wait_hint(kv_full_addr, _phase_kv_full_0, 10000000);
            _phase_kv_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (elect_sync()) {
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa0, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4))));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa0 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 24)));
                tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa1, make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 4), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 8)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 8), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 16)));
                tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa1 + 12), make_sf_cp_desc_lo_sbo512((((smem_ksf_addr) >> 4) + 128 + 24)));
                int _mma_a_lo_0 = ((smem_krope_addr) >> 4) & 0x3FFF;
                int _mma_b_lo_0 = ((smem_qrope_b_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 134743184;\n\t"
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
                int _mma_a_lo_1 = (((smem_kf4_addr) >> 4) & 0x3FFF) + (0) * 1024;
                int _mma_b_lo_1 = (((smem_qf4b_addr) >> 4) & 0x3FFF) + (0) * 1024;
                {
                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                        0x8080480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb0 + 0, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                        0x8080480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb0 + 4, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                        0x8080480U, tmem_tmem_sfa0 + 8, tmem_tmem_sfb0 + 8, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                        0x8080480U, tmem_tmem_sfa0 + 12, tmem_tmem_sfb0 + 12, 1);
                }
                int _mma_a_lo_2 = (((smem_kf4_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_2 = (((smem_qf4b_addr) >> 4) & 0x3FFF) + (1) * 1024;
                {
                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                        0x8080480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb1 + 0, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                        0x8080480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb1 + 4, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                        0x8080480U, tmem_tmem_sfa1 + 8, tmem_tmem_sfb1 + 8, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                        0x8080480U, tmem_tmem_sfa1 + 12, tmem_tmem_sfb1 + 12, 1);
                }
                tcgen05_commit(s_full_addr);
            }
            unsigned int _phase_p_full_0 = 0;
            mbarrier_wait_hint(p_full_addr, _phase_p_full_0, 10000000);
            _phase_p_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (elect_sync()) {
                int _mma_a_lo_3 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (0) * 1024;
                int _mma_b_lo_3 = (((smem_pb_addr) >> 4) & 0x3FFF) | 0x1000000;
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
                    "mov.b32 id, 134774800;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_tmem_o0), "r"(0));
                tcgen05_commit(o_full_addr);
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait_hint(tmem_dealloc_addr, _phase_tmem_dealloc_0, 10000000);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 13 && warp <= 15) {
        { // load_warp_main
            const int load_tid = (warp - 13) * 32 + lane;
            int work_idx_3 = blockIdx.x / 4;
            int head_tile_3 = work_idx_3 % num_head_tiles;
            int split_work_3 = work_idx_3 / num_head_tiles;
            int query_idx_3 = split_work_3 / num_splits;
            if (load_tid == 0) {
                mbarrier_arrive_expect_tx(q_nope_full0_addr, 12288);
                tma_4d_gmem2smem(smem_qstage_addr, (&tmap_q), 0, head_tile_3 * 128, 0, query_idx_3, q_nope_full0_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 4096, (&tmap_q), 0, head_tile_3 * 128, 1, query_idx_3, q_nope_full0_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 8192, (&tmap_q), 0, head_tile_3 * 128, 2, query_idx_3, q_nope_full0_addr);
                mbarrier_arrive_expect_tx(q_nope_full1_addr, 8192);
                tma_4d_gmem2smem(smem_qstage_addr + 12288, (&tmap_q), 0, head_tile_3 * 128, 3, query_idx_3, q_nope_full1_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 16384, (&tmap_q), 0, head_tile_3 * 128, 4, query_idx_3, q_nope_full1_addr);
                mbarrier_arrive_expect_tx(q_nope_full2_addr, 8192);
                tma_4d_gmem2smem(smem_qstage_addr + 20480, (&tmap_q), 0, head_tile_3 * 128, 5, query_idx_3, q_nope_full2_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 24576, (&tmap_q), 0, head_tile_3 * 128, 6, query_idx_3, q_nope_full2_addr);
                mbarrier_arrive_expect_tx(q_rope_full_addr, 4096);
                tma_4d_gmem2smem(smem_qrope_addr, (&tmap_q), 0, head_tile_3 * 128, 7, query_idx_3, q_rope_full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
