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
#define TMEM_NCOLS 144
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O0_OFFSET 16
#define TMEM_TMEM_O1_OFFSET 32
#define TMEM_TMEM_O2_OFFSET 48
#define TMEM_TMEM_O3_OFFSET 64
#define TMEM_TMEM_SFA0_OFFSET 80
#define TMEM_TMEM_SFA1_OFFSET 96
#define TMEM_TMEM_SFB0_OFFSET 112
#define TMEM_TMEM_SFB1_OFFSET 128
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
#define SMEM_SMEM_QROPE_STAGE_BYTES 2048
#define SMEM_SMEM_QROPE_STRIDE 2048
#define SMEM_SMEM_QSTAGE_OFF 173056
#define SMEM_SMEM_QSTAGE_STAGE_BYTES 2048
#define SMEM_SMEM_QSTAGE_STRIDE 2048
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
#define SMEM_SMEM_SFS_OFF 189440
#define SMEM_SMEM_SFS_STAGE_BYTES 4096
#define SMEM_SMEM_SFS_STRIDE 4096
#define SMEM_SMEM_P_OFF 173056
#define SMEM_SMEM_P_STAGE_BYTES 16384
#define SMEM_SMEM_P_STRIDE 16384
#define SMEM_SMEM_RCPTAB_OFF 193536
#define SMEM_SMEM_RCPTAB_STAGE_BYTES 64
#define SMEM_SMEM_RCPTAB_STRIDE 64
#define SMEM_SMEM_MASK_OFF 195584
#define SMEM_SMEM_MASK_STAGE_BYTES 16
#define SMEM_SMEM_MASK_STRIDE 16
#define SMEM_SMEM_TOK_OFF 195616
#define SMEM_SMEM_TOK_STAGE_BYTES 512
#define SMEM_SMEM_TOK_STRIDE 512
#define SMEM_SMEM_ROWOFF_OFF 197792
#define SMEM_SMEM_ROWOFF_STAGE_BYTES 2048
#define SMEM_SMEM_ROWOFF_STRIDE 2048
#define SMEM_SMEM_SFS32_OFF 189440
#define SMEM_SMEM_SFS32_STAGE_BYTES 4096
#define SMEM_SMEM_SFS32_STRIDE 4096
#define SMEM_SMEM_PMAX_OFF 196640
#define SMEM_SMEM_PMAX_STAGE_BYTES 384
#define SMEM_SMEM_PMAX_STRIDE 384
#define SMEM_SMEM_PSUM_OFF 197024
#define SMEM_SMEM_PSUM_STAGE_BYTES 384
#define SMEM_SMEM_PSUM_STRIDE 384
#define SMEM_SMEM_RSUM_OFF 197408
#define SMEM_SMEM_RSUM_STAGE_BYTES 384
#define SMEM_SMEM_RSUM_STRIDE 384
#define SMEM_SMEM_QF4B_OFF 1024
#define SMEM_SMEM_QF4B_STAGE_BYTES 2048
#define SMEM_SMEM_QF4B_STRIDE 16384
#define SMEM_SMEM_QROPE_B_OFF 37888
#define SMEM_SMEM_QROPE_B_STAGE_BYTES 2048
#define SMEM_SMEM_QROPE_B_STRIDE 2048
#define SMEM_SMEM_PB_OFF 173056
#define SMEM_SMEM_PB_STAGE_BYTES 2048
#define SMEM_SMEM_PB_STRIDE 2048
#define SMEM_TOTAL 199936
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
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

__global__ __launch_bounds__(512, 1) void
kernel_cake_dsv4_nvfp4_cf6c97a0107a1f15fc89(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_out, __nv_bfloat16* __restrict__ q_rows, uint8_t* __restrict__ main_cache, uint8_t* __restrict__ extra_cache, int* __restrict__ main_indices, int* __restrict__ extra_indices, int* __restrict__ main_lengths, int* __restrict__ extra_lengths, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ lse_out, int num_heads, int num_head_tiles, int num_splits, int num_main_tiles, int main_width, int extra_width, int main_index_stride, int extra_index_stride, int has_main_lengths, int has_extra_lengths, int main_page_shift, int extra_page_shift, long long main_page_stride, long long extra_page_stride, int has_sinks, float lse_partial_scale, float lse_scale)
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
    #define tmem_dealloc_addr (mbar_base + 96)

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
    __nv_bfloat16* smem_qstage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int smem_qstage_addr = smem + 173056;
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
    uint8_t* smem_sfs = reinterpret_cast<uint8_t*>(smem_raw + 189440);
    const int smem_sfs_addr = smem + 189440;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 173056);
    const int smem_p_addr = smem + 173056;
    unsigned int* smem_rcptab = reinterpret_cast<unsigned int*>(smem_raw + 193536);
    const int smem_rcptab_addr = smem + 193536;
    unsigned int* smem_mask = reinterpret_cast<unsigned int*>(smem_raw + 195584);
    const int smem_mask_addr = smem + 195584;
    int* smem_tok = reinterpret_cast<int*>(smem_raw + 195616);
    const int smem_tok_addr = smem + 195616;
    unsigned int* smem_rowoff = reinterpret_cast<unsigned int*>(smem_raw + 197792);
    const int smem_rowoff_addr = smem + 197792;
    unsigned int* smem_sfs32 = reinterpret_cast<unsigned int*>(smem_raw + 189440);
    const int smem_sfs32_addr = smem + 189440;
    float* smem_pmax = reinterpret_cast<float*>(smem_raw + 196640);
    const int smem_pmax_addr = smem + 196640;
    float* smem_psum = reinterpret_cast<float*>(smem_raw + 197024);
    const int smem_psum_addr = smem + 197024;
    float* smem_rsum = reinterpret_cast<float*>(smem_raw + 197408);
    const int smem_rsum_addr = smem + 197408;
    uint8_t* smem_qf4b = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_qf4b_addr = smem + 1024;
    __nv_bfloat16* smem_qrope_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 37888);
    const int smem_qrope_b_addr = smem + 37888;
    uint8_t* smem_pb = reinterpret_cast<uint8_t*>(smem_raw + 173056);
    const int smem_pb_addr = smem + 173056;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_out))) : "memory");

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 13 barriers)
    // Mbarriers at smem_raw[0..104)

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
            // o_full: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // tmem_dealloc: 1 barriers, init_count=384
            mbarrier_init(smem + 96, 384);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 144 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 104);
    if (warp == 0) {
        int _tmem_hold = smem + 104;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o0 = taddr + 16;
    const int tmem_tmem_o1 = taddr + 32;
    const int tmem_tmem_o2 = taddr + 48;
    const int tmem_tmem_o3 = taddr + 64;
    const int tmem_tmem_sfa0 = taddr + 80;
    const int tmem_tmem_sfa1 = taddr + 96;
    const int tmem_tmem_sfb0 = taddr + 112;
    const int tmem_tmem_sfb1 = taddr + 128;

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
            int o_chunk = 0;
            int work_idx = blockIdx.x;
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
                    if (head_base + q_block * 32 < num_heads && q_row < 16) {
                        int q_row_addr = smem_qstage_addr + (unsigned int)(kset * 2048) + (unsigned int)(q_row * 128);
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
                int vblock = 32 * o_chunk;
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
                int vblock_0 = 32 * o_chunk + 1;
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
                int vblock_2 = 32 * o_chunk + 2;
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
                int vblock_4 = 32 * o_chunk + 3;
                unsigned int v8_5[4];
                {
                    int vchunk_3 = vblock_4 >> 1;
                    int vhalf_3 = vblock_4 & 1;
                    unsigned int kraw_3[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_3[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_3 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_3 % 8 * 16 ^ row % 8 * 16)) + (unsigned int)(8 * vhalf_3)));
                    unsigned int sfw32_3 = smem_sfs32[row * 32 + vblock_4 >> 2];
                    unsigned int scale_3 = sfw32_3 >> (unsigned int)(8 * (vblock_4 & 3)) & 255;
                    {
                        v8_5[0] = cake_dsv4_qmul4_portable<5>(kraw_3[0], scale_3);
                    }
                    {
                        v8_5[1] = cake_dsv4_qmul4_portable<6>(kraw_3[0], scale_3);
                    }
                    {
                        v8_5[2] = cake_dsv4_qmul4_portable<5>(kraw_3[1], scale_3);
                    }
                    {
                        v8_5[3] = cake_dsv4_qmul4_portable<6>(kraw_3[1], scale_3);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row * 128 + (48 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_5[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5[(0) + 3])));
                int vblock_6 = 32 * o_chunk + 4;
                unsigned int v8_7[4];
                {
                    int vchunk_4 = vblock_6 >> 1;
                    int vhalf_4 = vblock_6 & 1;
                    unsigned int kraw_4[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_4[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_4 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_4 % 8 * 16 ^ row % 8 * 16)) + (unsigned int)(8 * vhalf_4)));
                    unsigned int sfw32_4 = smem_sfs32[row * 32 + vblock_6 >> 2];
                    unsigned int scale_4 = sfw32_4 >> (unsigned int)(8 * (vblock_6 & 3)) & 255;
                    {
                        v8_7[0] = cake_dsv4_qmul4_portable<5>(kraw_4[0], scale_4);
                    }
                    {
                        v8_7[1] = cake_dsv4_qmul4_portable<6>(kraw_4[0], scale_4);
                    }
                    {
                        v8_7[2] = cake_dsv4_qmul4_portable<5>(kraw_4[1], scale_4);
                    }
                    {
                        v8_7[3] = cake_dsv4_qmul4_portable<6>(kraw_4[1], scale_4);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row * 128 + (64 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_7[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7[(0) + 3])));
                int vblock_8 = 32 * o_chunk + 5;
                unsigned int v8_9[4];
                {
                    int vchunk_5 = vblock_8 >> 1;
                    int vhalf_5 = vblock_8 & 1;
                    unsigned int kraw_5[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_5[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_5 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_5 % 8 * 16 ^ row % 8 * 16)) + (unsigned int)(8 * vhalf_5)));
                    unsigned int sfw32_5 = smem_sfs32[row * 32 + vblock_8 >> 2];
                    unsigned int scale_5 = sfw32_5 >> (unsigned int)(8 * (vblock_8 & 3)) & 255;
                    {
                        v8_9[0] = cake_dsv4_qmul4_portable<5>(kraw_5[0], scale_5);
                    }
                    {
                        v8_9[1] = cake_dsv4_qmul4_portable<6>(kraw_5[0], scale_5);
                    }
                    {
                        v8_9[2] = cake_dsv4_qmul4_portable<5>(kraw_5[1], scale_5);
                    }
                    {
                        v8_9[3] = cake_dsv4_qmul4_portable<6>(kraw_5[1], scale_5);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row * 128 + (80 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_9[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_9[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_9[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_9[(0) + 3])));
                int vblock_10 = 32 * o_chunk + 6;
                unsigned int v8_11[4];
                {
                    int vchunk_6 = vblock_10 >> 1;
                    int vhalf_6 = vblock_10 & 1;
                    unsigned int kraw_6[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_6[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_6 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_6 % 8 * 16 ^ row % 8 * 16)) + (unsigned int)(8 * vhalf_6)));
                    unsigned int sfw32_6 = smem_sfs32[row * 32 + vblock_10 >> 2];
                    unsigned int scale_6 = sfw32_6 >> (unsigned int)(8 * (vblock_10 & 3)) & 255;
                    {
                        v8_11[0] = cake_dsv4_qmul4_portable<5>(kraw_6[0], scale_6);
                    }
                    {
                        v8_11[1] = cake_dsv4_qmul4_portable<6>(kraw_6[0], scale_6);
                    }
                    {
                        v8_11[2] = cake_dsv4_qmul4_portable<5>(kraw_6[1], scale_6);
                    }
                    {
                        v8_11[3] = cake_dsv4_qmul4_portable<6>(kraw_6[1], scale_6);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row * 128 + (96 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_11[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11[(0) + 3])));
                int vblock_12 = 32 * o_chunk + 7;
                unsigned int v8_13[4];
                {
                    int vchunk_7 = vblock_12 >> 1;
                    int vhalf_7 = vblock_12 & 1;
                    unsigned int kraw_7[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_7[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_7 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_7 % 8 * 16 ^ row % 8 * 16)) + (unsigned int)(8 * vhalf_7)));
                    unsigned int sfw32_7 = smem_sfs32[row * 32 + vblock_12 >> 2];
                    unsigned int scale_7 = sfw32_7 >> (unsigned int)(8 * (vblock_12 & 3)) & 255;
                    {
                        v8_13[0] = cake_dsv4_qmul4_portable<5>(kraw_7[0], scale_7);
                    }
                    {
                        v8_13[1] = cake_dsv4_qmul4_portable<6>(kraw_7[0], scale_7);
                    }
                    {
                        v8_13[2] = cake_dsv4_qmul4_portable<5>(kraw_7[1], scale_7);
                    }
                    {
                        v8_13[3] = cake_dsv4_qmul4_portable<6>(kraw_7[1], scale_7);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(row * 128 + (112 ^ row % 8 * 16))), "r"(*reinterpret_cast<uint32_t*>(&v8_13[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_13[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_13[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_13[(0) + 3])));
                int vblock_14 = 32 * o_chunk + 8;
                unsigned int v8_15[4];
                {
                    int vchunk_8 = vblock_14 >> 1;
                    int vhalf_8 = vblock_14 & 1;
                    unsigned int kraw_8[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_8[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_8 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_8 % 8 * 16 ^ row % 8 * 16)) + (unsigned int)(8 * vhalf_8)));
                    unsigned int sfw32_8 = smem_sfs32[row * 32 + vblock_14 >> 2];
                    unsigned int scale_8 = sfw32_8 >> (unsigned int)(8 * (vblock_14 & 3)) & 255;
                    {
                        v8_15[0] = cake_dsv4_qmul4_portable<5>(kraw_8[0], scale_8);
                    }
                    {
                        v8_15[1] = cake_dsv4_qmul4_portable<6>(kraw_8[0], scale_8);
                    }
                    {
                        v8_15[2] = cake_dsv4_qmul4_portable<5>(kraw_8[1], scale_8);
                    }
                    {
                        v8_15[3] = cake_dsv4_qmul4_portable<6>(kraw_8[1], scale_8);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (0 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15[(0) + 3])));
                int vblock_16 = 32 * o_chunk + 9;
                unsigned int v8_17[4];
                {
                    int vchunk_9 = vblock_16 >> 1;
                    int vhalf_9 = vblock_16 & 1;
                    unsigned int kraw_9[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_9[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_9 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_9 % 8 * 16 ^ row % 8 * 16)) + (unsigned int)(8 * vhalf_9)));
                    unsigned int sfw32_9 = smem_sfs32[row * 32 + vblock_16 >> 2];
                    unsigned int scale_9 = sfw32_9 >> (unsigned int)(8 * (vblock_16 & 3)) & 255;
                    {
                        v8_17[0] = cake_dsv4_qmul4_portable<5>(kraw_9[0], scale_9);
                    }
                    {
                        v8_17[1] = cake_dsv4_qmul4_portable<6>(kraw_9[0], scale_9);
                    }
                    {
                        v8_17[2] = cake_dsv4_qmul4_portable<5>(kraw_9[1], scale_9);
                    }
                    {
                        v8_17[3] = cake_dsv4_qmul4_portable<6>(kraw_9[1], scale_9);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (16 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_17[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_17[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_17[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_17[(0) + 3])));
                int vblock_18 = 32 * o_chunk + 10;
                unsigned int v8_19[4];
                {
                    int vchunk_10 = vblock_18 >> 1;
                    int vhalf_10 = vblock_18 & 1;
                    unsigned int kraw_10[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_10[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_10 / 8 * 16384) + (unsigned int)(row * 128 + (vchunk_10 % 8 * 16 ^ row % 8 * 16)) + (unsigned int)(8 * vhalf_10)));
                    unsigned int sfw32_10 = smem_sfs32[row * 32 + vblock_18 >> 2];
                    unsigned int scale_10 = sfw32_10 >> (unsigned int)(8 * (vblock_18 & 3)) & 255;
                    {
                        v8_19[0] = cake_dsv4_qmul4_portable<5>(kraw_10[0], scale_10);
                    }
                    {
                        v8_19[1] = cake_dsv4_qmul4_portable<6>(kraw_10[0], scale_10);
                    }
                    {
                        v8_19[2] = cake_dsv4_qmul4_portable<5>(kraw_10[1], scale_10);
                    }
                    {
                        v8_19[3] = cake_dsv4_qmul4_portable<6>(kraw_10[1], scale_10);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (32 ^ row % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_19[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_19[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_19[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_19[(0) + 3])));
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
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (48 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (64 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (80 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (96 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(row * 128 + (112 ^ row % 8 * 16))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (0 ^ row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (16 ^ row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row * 128 + (32 ^ row % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            unsigned int _phase_o_full_2 = 0;
            unsigned int _phase_o_full_3 = 0;
            {
                mbarrier_wait_hint(s_full_addr, _phase_s_full_0, 10000000);
                _phase_s_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float score_values[8];
                tmem_ld_x8(&score_values[0], taddr + (unsigned int)(tmem_row_origin << 16));
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
                }
                float tr_vals[8];
                tr_vals[0] = score_values[0];
                tr_vals[1] = score_values[1];
                tr_vals[2] = score_values[2];
                tr_vals[3] = score_values[3];
                tr_vals[4] = score_values[4];
                tr_vals[5] = score_values[5];
                tr_vals[6] = score_values[6];
                tr_vals[7] = score_values[7];
                int hi_bit = lane & 16;
                float send = ((hi_bit != 0) ? tr_vals[0] : tr_vals[4]);
                float keep = ((hi_bit != 0) ? tr_vals[4] : tr_vals[0]);
                float _shfl_0;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(send), "r"(lane ^ 16));
                float recv = _shfl_0;
                float _max_30 = max_noftz(keep, recv);
                tr_vals[0] = _max_30;
                float send_0 = ((hi_bit != 0) ? tr_vals[1] : tr_vals[5]);
                float keep_1 = ((hi_bit != 0) ? tr_vals[5] : tr_vals[1]);
                float _shfl_1;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_1) : "f"(send_0), "r"(lane ^ 16));
                float recv_2 = _shfl_1;
                float _max_31 = max_noftz(keep_1, recv_2);
                tr_vals[1] = _max_31;
                float send_3 = ((hi_bit != 0) ? tr_vals[2] : tr_vals[6]);
                float keep_4 = ((hi_bit != 0) ? tr_vals[6] : tr_vals[2]);
                float _shfl_2;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_2) : "f"(send_3), "r"(lane ^ 16));
                float recv_5 = _shfl_2;
                float _max_32 = max_noftz(keep_4, recv_5);
                tr_vals[2] = _max_32;
                float send_6 = ((hi_bit != 0) ? tr_vals[3] : tr_vals[7]);
                float keep_7 = ((hi_bit != 0) ? tr_vals[7] : tr_vals[3]);
                float _shfl_3;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_3) : "f"(send_6), "r"(lane ^ 16));
                float recv_8 = _shfl_3;
                float _max_33 = max_noftz(keep_7, recv_8);
                tr_vals[3] = _max_33;
                int hi_bit_9 = lane & 8;
                float send_10 = ((hi_bit_9 != 0) ? tr_vals[0] : tr_vals[2]);
                float keep_11 = ((hi_bit_9 != 0) ? tr_vals[2] : tr_vals[0]);
                float _shfl_4;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_4) : "f"(send_10), "r"(lane ^ 8));
                float recv_12 = _shfl_4;
                float _max_34 = max_noftz(keep_11, recv_12);
                tr_vals[0] = _max_34;
                float send_13 = ((hi_bit_9 != 0) ? tr_vals[1] : tr_vals[3]);
                float keep_14 = ((hi_bit_9 != 0) ? tr_vals[3] : tr_vals[1]);
                float _shfl_5;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_5) : "f"(send_13), "r"(lane ^ 8));
                float recv_15 = _shfl_5;
                float _max_35 = max_noftz(keep_14, recv_15);
                tr_vals[1] = _max_35;
                int hi_bit_16 = lane & 4;
                float send_17 = ((hi_bit_16 != 0) ? tr_vals[0] : tr_vals[1]);
                float keep_18 = ((hi_bit_16 != 0) ? tr_vals[1] : tr_vals[0]);
                float _shfl_6;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_6) : "f"(send_17), "r"(lane ^ 4));
                float recv_19 = _shfl_6;
                float _max_36 = max_noftz(keep_18, recv_19);
                tr_vals[0] = _max_36;
                float _shfl_7;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_7) : "f"(tr_vals[0]), "r"(lane ^ 2));
                float other = _shfl_7;
                float _max_37 = max_noftz(tr_vals[0], other);
                tr_vals[0] = _max_37;
                float _shfl_8;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_8) : "f"(tr_vals[0]), "r"(lane ^ 1));
                float other_20 = _shfl_8;
                float _max_38 = max_noftz(tr_vals[0], other_20);
                tr_vals[0] = _max_38;
                if ((lane & 3) == 0) {
                    smem_pmax[local_warp * 8 + (lane >> 2)] = tr_vals[0];
                }
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                float m_lane = -CAKE_INF;
                float sink_lane = -CAKE_INF;
                float m_scaled = 0.0f;
                if (lane < 8) {
                    float _max_39 = max_noftz(smem_pmax[lane], smem_pmax[8 + lane]);
                    float _max_40 = max_noftz(smem_pmax[16 + lane], smem_pmax[24 + lane]);
                    float _max_41 = max_noftz(_max_39, _max_40);
                    m_lane = _max_41;
                    if (has_sinks != 0 && split_idx == 0 && head_base + lane < num_heads) {
                        sink_lane = sinks[head_base + lane] * 1.4426950408889634f;
                    }
                    float _max_42 = max_noftz(m_lane * softmax_scale_log2, sink_lane);
                    m_scaled = _max_42;
                    if (m_scaled == -CAKE_INF) {
                        m_scaled = 0.0f;
                    }
                }
                float col_max[8];
                float _shfl_9;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_9) : "f"(m_scaled), "r"(0));
                col_max[0] = _shfl_9;
                float _shfl_10;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_10) : "f"(m_scaled), "r"(1));
                col_max[1] = _shfl_10;
                float _shfl_11;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_11) : "f"(m_scaled), "r"(2));
                col_max[2] = _shfl_11;
                float _shfl_12;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_12) : "f"(m_scaled), "r"(3));
                col_max[3] = _shfl_12;
                float _shfl_13;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_13) : "f"(m_scaled), "r"(4));
                col_max[4] = _shfl_13;
                float _shfl_14;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_14) : "f"(m_scaled), "r"(5));
                col_max[5] = _shfl_14;
                float _shfl_15;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_15) : "f"(m_scaled), "r"(6));
                col_max[6] = _shfl_15;
                float _shfl_16;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_16) : "f"(m_scaled), "r"(7));
                col_max[7] = _shfl_16;
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
                {
                    uint16_t _fp8_pair_46;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_46) : "f"(0.0f), "f"(score_values[0]));
                    uint32_t _byte_46 = (uint32_t)(_fp8_pair_46 & 0xFF);
                    uint32_t _addr_46 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(row ^ (row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_46), "r"(_byte_46) : "memory");
                }
                float _fp8_rt_0;
                uint16_t _e4m3x2_47;
                uint32_t _f16x2_47;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_47) : "f"(0.0f), "f"(score_values[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_47) : "h"(_e4m3x2_47));
                uint16_t _fp8_h0_47 = (uint16_t)(_f16x2_47 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_47));
                tr_vals[0] = _fp8_rt_0;
                {
                    uint16_t _fp8_pair_48;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_48) : "f"(0.0f), "f"(score_values[1]));
                    uint32_t _byte_48 = (uint32_t)(_fp8_pair_48 & 0xFF);
                    uint32_t _addr_48 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(128 + row ^ (128 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_48), "r"(_byte_48) : "memory");
                }
                float _fp8_rt_1;
                uint16_t _e4m3x2_49;
                uint32_t _f16x2_49;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_49) : "f"(0.0f), "f"(score_values[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_49) : "h"(_e4m3x2_49));
                uint16_t _fp8_h0_49 = (uint16_t)(_f16x2_49 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_49));
                tr_vals[1] = _fp8_rt_1;
                {
                    uint16_t _fp8_pair_50;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_50) : "f"(0.0f), "f"(score_values[2]));
                    uint32_t _byte_50 = (uint32_t)(_fp8_pair_50 & 0xFF);
                    uint32_t _addr_50 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(256 + row ^ (256 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_50), "r"(_byte_50) : "memory");
                }
                float _fp8_rt_2;
                uint16_t _e4m3x2_51;
                uint32_t _f16x2_51;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_51) : "f"(0.0f), "f"(score_values[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_51) : "h"(_e4m3x2_51));
                uint16_t _fp8_h0_51 = (uint16_t)(_f16x2_51 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_2) : "h"(_fp8_h0_51));
                tr_vals[2] = _fp8_rt_2;
                {
                    uint16_t _fp8_pair_52;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_52) : "f"(0.0f), "f"(score_values[3]));
                    uint32_t _byte_52 = (uint32_t)(_fp8_pair_52 & 0xFF);
                    uint32_t _addr_52 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(384 + row ^ (384 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_52), "r"(_byte_52) : "memory");
                }
                float _fp8_rt_3;
                uint16_t _e4m3x2_53;
                uint32_t _f16x2_53;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_53) : "f"(0.0f), "f"(score_values[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_53) : "h"(_e4m3x2_53));
                uint16_t _fp8_h0_53 = (uint16_t)(_f16x2_53 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_3) : "h"(_fp8_h0_53));
                tr_vals[3] = _fp8_rt_3;
                {
                    uint16_t _fp8_pair_54;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_54) : "f"(0.0f), "f"(score_values[4]));
                    uint32_t _byte_54 = (uint32_t)(_fp8_pair_54 & 0xFF);
                    uint32_t _addr_54 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(512 + row ^ (512 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_54), "r"(_byte_54) : "memory");
                }
                float _fp8_rt_4;
                uint16_t _e4m3x2_55;
                uint32_t _f16x2_55;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_55) : "f"(0.0f), "f"(score_values[4]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_55) : "h"(_e4m3x2_55));
                uint16_t _fp8_h0_55 = (uint16_t)(_f16x2_55 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_4) : "h"(_fp8_h0_55));
                tr_vals[4] = _fp8_rt_4;
                {
                    uint16_t _fp8_pair_56;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_56) : "f"(0.0f), "f"(score_values[5]));
                    uint32_t _byte_56 = (uint32_t)(_fp8_pair_56 & 0xFF);
                    uint32_t _addr_56 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(640 + row ^ (640 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_56), "r"(_byte_56) : "memory");
                }
                float _fp8_rt_5;
                uint16_t _e4m3x2_57;
                uint32_t _f16x2_57;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_57) : "f"(0.0f), "f"(score_values[5]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_57) : "h"(_e4m3x2_57));
                uint16_t _fp8_h0_57 = (uint16_t)(_f16x2_57 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_5) : "h"(_fp8_h0_57));
                tr_vals[5] = _fp8_rt_5;
                {
                    uint16_t _fp8_pair_58;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_58) : "f"(0.0f), "f"(score_values[6]));
                    uint32_t _byte_58 = (uint32_t)(_fp8_pair_58 & 0xFF);
                    uint32_t _addr_58 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(768 + row ^ (768 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_58), "r"(_byte_58) : "memory");
                }
                float _fp8_rt_6;
                uint16_t _e4m3x2_59;
                uint32_t _f16x2_59;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_59) : "f"(0.0f), "f"(score_values[6]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_59) : "h"(_e4m3x2_59));
                uint16_t _fp8_h0_59 = (uint16_t)(_f16x2_59 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_6) : "h"(_fp8_h0_59));
                tr_vals[6] = _fp8_rt_6;
                {
                    uint16_t _fp8_pair_60;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_60) : "f"(0.0f), "f"(score_values[7]));
                    uint32_t _byte_60 = (uint32_t)(_fp8_pair_60 & 0xFF);
                    uint32_t _addr_60 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(896 + row ^ (896 + row >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_60), "r"(_byte_60) : "memory");
                }
                float _fp8_rt_7;
                uint16_t _e4m3x2_61;
                uint32_t _f16x2_61;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_61) : "f"(0.0f), "f"(score_values[7]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_61) : "h"(_e4m3x2_61));
                uint16_t _fp8_h0_61 = (uint16_t)(_f16x2_61 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_7) : "h"(_fp8_h0_61));
                tr_vals[7] = _fp8_rt_7;
                int hi_bit_21 = lane & 16;
                float send_22 = ((hi_bit_21 != 0) ? score_values[0] : score_values[4]);
                float keep_23 = ((hi_bit_21 != 0) ? score_values[4] : score_values[0]);
                float _shfl_17;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_17) : "f"(send_22), "r"(lane ^ 16));
                float recv_24 = _shfl_17;
                score_values[0] = keep_23 + recv_24;
                float send_25 = ((hi_bit_21 != 0) ? score_values[1] : score_values[5]);
                float keep_26 = ((hi_bit_21 != 0) ? score_values[5] : score_values[1]);
                float _shfl_18;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_18) : "f"(send_25), "r"(lane ^ 16));
                float recv_27 = _shfl_18;
                score_values[1] = keep_26 + recv_27;
                float send_28 = ((hi_bit_21 != 0) ? score_values[2] : score_values[6]);
                float keep_29 = ((hi_bit_21 != 0) ? score_values[6] : score_values[2]);
                float _shfl_19;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_19) : "f"(send_28), "r"(lane ^ 16));
                float recv_30 = _shfl_19;
                score_values[2] = keep_29 + recv_30;
                float send_31 = ((hi_bit_21 != 0) ? score_values[3] : score_values[7]);
                float keep_32 = ((hi_bit_21 != 0) ? score_values[7] : score_values[3]);
                float _shfl_20;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_20) : "f"(send_31), "r"(lane ^ 16));
                float recv_33 = _shfl_20;
                score_values[3] = keep_32 + recv_33;
                int hi_bit_34 = lane & 8;
                float send_35 = ((hi_bit_34 != 0) ? score_values[0] : score_values[2]);
                float keep_36 = ((hi_bit_34 != 0) ? score_values[2] : score_values[0]);
                float _shfl_21;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_21) : "f"(send_35), "r"(lane ^ 8));
                float recv_37 = _shfl_21;
                score_values[0] = keep_36 + recv_37;
                float send_38 = ((hi_bit_34 != 0) ? score_values[1] : score_values[3]);
                float keep_39 = ((hi_bit_34 != 0) ? score_values[3] : score_values[1]);
                float _shfl_22;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_22) : "f"(send_38), "r"(lane ^ 8));
                float recv_40 = _shfl_22;
                score_values[1] = keep_39 + recv_40;
                int hi_bit_41 = lane & 4;
                float send_42 = ((hi_bit_41 != 0) ? score_values[0] : score_values[1]);
                float keep_43 = ((hi_bit_41 != 0) ? score_values[1] : score_values[0]);
                float _shfl_23;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_23) : "f"(send_42), "r"(lane ^ 4));
                float recv_44 = _shfl_23;
                score_values[0] = keep_43 + recv_44;
                float _shfl_24;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_24) : "f"(score_values[0]), "r"(lane ^ 2));
                float other_45 = _shfl_24;
                score_values[0] = score_values[0] + other_45;
                float _shfl_25;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_25) : "f"(score_values[0]), "r"(lane ^ 1));
                float other_46 = _shfl_25;
                score_values[0] = score_values[0] + other_46;
                int hi_bit_47 = lane & 16;
                float send_48 = ((hi_bit_47 != 0) ? tr_vals[0] : tr_vals[4]);
                float keep_49 = ((hi_bit_47 != 0) ? tr_vals[4] : tr_vals[0]);
                float _shfl_26;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_26) : "f"(send_48), "r"(lane ^ 16));
                float recv_50 = _shfl_26;
                tr_vals[0] = keep_49 + recv_50;
                float send_51 = ((hi_bit_47 != 0) ? tr_vals[1] : tr_vals[5]);
                float keep_52 = ((hi_bit_47 != 0) ? tr_vals[5] : tr_vals[1]);
                float _shfl_27;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_27) : "f"(send_51), "r"(lane ^ 16));
                float recv_53 = _shfl_27;
                tr_vals[1] = keep_52 + recv_53;
                float send_54 = ((hi_bit_47 != 0) ? tr_vals[2] : tr_vals[6]);
                float keep_55 = ((hi_bit_47 != 0) ? tr_vals[6] : tr_vals[2]);
                float _shfl_28;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_28) : "f"(send_54), "r"(lane ^ 16));
                float recv_56 = _shfl_28;
                tr_vals[2] = keep_55 + recv_56;
                float send_57 = ((hi_bit_47 != 0) ? tr_vals[3] : tr_vals[7]);
                float keep_58 = ((hi_bit_47 != 0) ? tr_vals[7] : tr_vals[3]);
                float _shfl_29;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_29) : "f"(send_57), "r"(lane ^ 16));
                float recv_59 = _shfl_29;
                tr_vals[3] = keep_58 + recv_59;
                int hi_bit_60 = lane & 8;
                float send_61 = ((hi_bit_60 != 0) ? tr_vals[0] : tr_vals[2]);
                float keep_62 = ((hi_bit_60 != 0) ? tr_vals[2] : tr_vals[0]);
                float _shfl_30;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_30) : "f"(send_61), "r"(lane ^ 8));
                float recv_63 = _shfl_30;
                tr_vals[0] = keep_62 + recv_63;
                float send_64 = ((hi_bit_60 != 0) ? tr_vals[1] : tr_vals[3]);
                float keep_65 = ((hi_bit_60 != 0) ? tr_vals[3] : tr_vals[1]);
                float _shfl_31;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_31) : "f"(send_64), "r"(lane ^ 8));
                float recv_66 = _shfl_31;
                tr_vals[1] = keep_65 + recv_66;
                int hi_bit_67 = lane & 4;
                float send_68 = ((hi_bit_67 != 0) ? tr_vals[0] : tr_vals[1]);
                float keep_69 = ((hi_bit_67 != 0) ? tr_vals[1] : tr_vals[0]);
                float _shfl_32;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_32) : "f"(send_68), "r"(lane ^ 4));
                float recv_70 = _shfl_32;
                tr_vals[0] = keep_69 + recv_70;
                float _shfl_33;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_33) : "f"(tr_vals[0]), "r"(lane ^ 2));
                float other_71 = _shfl_33;
                tr_vals[0] = tr_vals[0] + other_71;
                float _shfl_34;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_34) : "f"(tr_vals[0]), "r"(lane ^ 1));
                float other_72 = _shfl_34;
                tr_vals[0] = tr_vals[0] + other_72;
                if ((lane & 3) == 0) {
                    smem_psum[local_warp * 8 + (lane >> 2)] = score_values[0];
                    smem_rsum[local_warp * 8 + (lane >> 2)] = tr_vals[0];
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                float norm_lane = 0.0f;
                if (lane < 8) {
                    float _exp2_8 = approx_exp2(sink_lane - m_scaled);
                    float sink_term = _exp2_8;
                    float col_sum = smem_psum[lane] + smem_psum[8 + lane] + smem_psum[16 + lane] + smem_psum[24 + lane] + sink_term;
                    float denom = smem_rsum[lane] + smem_rsum[8 + lane] + smem_rsum[16 + lane] + smem_rsum[24 + lane] + sink_term;
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
                float norm_c[8];
                float _shfl_35;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_35) : "f"(norm_lane), "r"(0));
                norm_c[0] = _shfl_35;
                float _shfl_36;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_36) : "f"(norm_lane), "r"(1));
                norm_c[1] = _shfl_36;
                float _shfl_37;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_37) : "f"(norm_lane), "r"(2));
                norm_c[2] = _shfl_37;
                float _shfl_38;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_38) : "f"(norm_lane), "r"(3));
                norm_c[3] = _shfl_38;
                float _shfl_39;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_39) : "f"(norm_lane), "r"(4));
                norm_c[4] = _shfl_39;
                float _shfl_40;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_40) : "f"(norm_lane), "r"(5));
                norm_c[5] = _shfl_40;
                float _shfl_41;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_41) : "f"(norm_lane), "r"(6));
                norm_c[6] = _shfl_41;
                float _shfl_42;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_42) : "f"(norm_lane), "r"(7));
                norm_c[7] = _shfl_42;
                float o_values[8];
                int dim = 0;
                long long out_off = 0;
                mbarrier_wait_hint(o_full_addr, _phase_o_full_0, 10000000);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x8(&o_values[0], taddr + 16 + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim = o_chunk * 4 * 128 + row;
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
                mbarrier_wait_hint(o_full_addr + 8, _phase_o_full_1, 10000000);
                _phase_o_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x8(&o_values[0], taddr + 32 + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim = (o_chunk * 4 + 1) * 128 + row;
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
                mbarrier_wait_hint(o_full_addr + 16, _phase_o_full_2, 10000000);
                _phase_o_full_2 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x8(&o_values[0], taddr + 48 + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim = (o_chunk * 4 + 2) * 128 + row;
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
                mbarrier_wait_hint(o_full_addr + 24, _phase_o_full_3, 10000000);
                _phase_o_full_3 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x8(&o_values[0], taddr + 64 + (unsigned int)(tmem_row_origin << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim = (o_chunk * 4 + 3) * 128 + row;
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
                mbarrier_arrive(tmem_dealloc_addr);
            }
        }
    }
    // ---- Role: compute1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // compute1_main
            const int local_warp_1 = warp - 4;
            int o_chunk_1 = 0;
            int work_idx_1 = blockIdx.x;
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
                    if (head_base_1 + q_block_1 * 32 < num_heads && q_row_1 < 16) {
                        int q_row_addr_1 = smem_qstage_addr + (unsigned int)(kset_1 * 2048) + (unsigned int)(q_row_1 * 128);
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
                            float _max_43 = max_noftz(_fabs_32, _fabs_33);
                            m8_1[0] = _max_43;
                            float _fabs_34 = fabsf(qv_1[2]);
                            float _fabs_35 = fabsf(qv_1[3]);
                            float _max_44 = max_noftz(_fabs_34, _fabs_35);
                            m8_1[1] = _max_44;
                            float _fabs_36 = fabsf(qv_1[4]);
                            float _fabs_37 = fabsf(qv_1[5]);
                            float _max_45 = max_noftz(_fabs_36, _fabs_37);
                            m8_1[2] = _max_45;
                            float _fabs_38 = fabsf(qv_1[6]);
                            float _fabs_39 = fabsf(qv_1[7]);
                            float _max_46 = max_noftz(_fabs_38, _fabs_39);
                            m8_1[3] = _max_46;
                            float _fabs_40 = fabsf(qv_1[8]);
                            float _fabs_41 = fabsf(qv_1[9]);
                            float _max_47 = max_noftz(_fabs_40, _fabs_41);
                            m8_1[4] = _max_47;
                            float _fabs_42 = fabsf(qv_1[10]);
                            float _fabs_43 = fabsf(qv_1[11]);
                            float _max_48 = max_noftz(_fabs_42, _fabs_43);
                            m8_1[5] = _max_48;
                            float _fabs_44 = fabsf(qv_1[12]);
                            float _fabs_45 = fabsf(qv_1[13]);
                            float _max_49 = max_noftz(_fabs_44, _fabs_45);
                            m8_1[6] = _max_49;
                            float _fabs_46 = fabsf(qv_1[14]);
                            float _fabs_47 = fabsf(qv_1[15]);
                            float _max_50 = max_noftz(_fabs_46, _fabs_47);
                            m8_1[7] = _max_50;
                            float m4_1[4];
                            float _max_51 = max_noftz(m8_1[0], m8_1[1]);
                            m4_1[0] = _max_51;
                            float _max_52 = max_noftz(m8_1[2], m8_1[3]);
                            m4_1[1] = _max_52;
                            float _max_53 = max_noftz(m8_1[4], m8_1[5]);
                            m4_1[2] = _max_53;
                            float _max_54 = max_noftz(m8_1[6], m8_1[7]);
                            m4_1[3] = _max_54;
                            float _max_55 = max_noftz(m4_1[0], m4_1[1]);
                            float _max_56 = max_noftz(m4_1[2], m4_1[3]);
                            float _max_57 = max_noftz(_max_55, _max_56);
                            float amax_1 = _max_57;
                            float sc_1 = amax_1 * inv_six_1;
                            uint16_t _e4m3x2_f32_178;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_178) : "f"(0.0f), "f"(sc_1));
                            uint16_t sc_pair_1 = _e4m3x2_f32_178;
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
                            float _max_58 = max_noftz(_fabs_48, _fabs_49);
                            m8_3_1[0] = _max_58;
                            float _fabs_50 = fabsf(qv_2_1[2]);
                            float _fabs_51 = fabsf(qv_2_1[3]);
                            float _max_59 = max_noftz(_fabs_50, _fabs_51);
                            m8_3_1[1] = _max_59;
                            float _fabs_52 = fabsf(qv_2_1[4]);
                            float _fabs_53 = fabsf(qv_2_1[5]);
                            float _max_60 = max_noftz(_fabs_52, _fabs_53);
                            m8_3_1[2] = _max_60;
                            float _fabs_54 = fabsf(qv_2_1[6]);
                            float _fabs_55 = fabsf(qv_2_1[7]);
                            float _max_61 = max_noftz(_fabs_54, _fabs_55);
                            m8_3_1[3] = _max_61;
                            float _fabs_56 = fabsf(qv_2_1[8]);
                            float _fabs_57 = fabsf(qv_2_1[9]);
                            float _max_62 = max_noftz(_fabs_56, _fabs_57);
                            m8_3_1[4] = _max_62;
                            float _fabs_58 = fabsf(qv_2_1[10]);
                            float _fabs_59 = fabsf(qv_2_1[11]);
                            float _max_63 = max_noftz(_fabs_58, _fabs_59);
                            m8_3_1[5] = _max_63;
                            float _fabs_60 = fabsf(qv_2_1[12]);
                            float _fabs_61 = fabsf(qv_2_1[13]);
                            float _max_64 = max_noftz(_fabs_60, _fabs_61);
                            m8_3_1[6] = _max_64;
                            float _fabs_62 = fabsf(qv_2_1[14]);
                            float _fabs_63 = fabsf(qv_2_1[15]);
                            float _max_65 = max_noftz(_fabs_62, _fabs_63);
                            m8_3_1[7] = _max_65;
                            float m4_4_1[4];
                            float _max_66 = max_noftz(m8_3_1[0], m8_3_1[1]);
                            m4_4_1[0] = _max_66;
                            float _max_67 = max_noftz(m8_3_1[2], m8_3_1[3]);
                            m4_4_1[1] = _max_67;
                            float _max_68 = max_noftz(m8_3_1[4], m8_3_1[5]);
                            m4_4_1[2] = _max_68;
                            float _max_69 = max_noftz(m8_3_1[6], m8_3_1[7]);
                            m4_4_1[3] = _max_69;
                            float _max_70 = max_noftz(m4_4_1[0], m4_4_1[1]);
                            float _max_71 = max_noftz(m4_4_1[2], m4_4_1[3]);
                            float _max_72 = max_noftz(_max_70, _max_71);
                            float amax_5_1 = _max_72;
                            float sc_6_1 = amax_5_1 * inv_six_1;
                            uint16_t _e4m3x2_f32_179;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_179) : "f"(0.0f), "f"(sc_6_1));
                            uint16_t sc_pair_7_1 = _e4m3x2_f32_179;
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
                int vblock_1 = 32 * o_chunk_1 + 11;
                unsigned int v8_2[4];
                {
                    int vchunk_11 = vblock_1 >> 1;
                    int vhalf_11 = vblock_1 & 1;
                    unsigned int kraw_11[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_11[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_11 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_11 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_11)));
                    unsigned int sfw32_11 = smem_sfs32[row_1 * 32 + vblock_1 >> 2];
                    unsigned int scale_11 = sfw32_11 >> (unsigned int)(8 * (vblock_1 & 3)) & 255;
                    {
                        v8_2[0] = cake_dsv4_qmul4_portable<5>(kraw_11[0], scale_11);
                    }
                    {
                        v8_2[1] = cake_dsv4_qmul4_portable<6>(kraw_11[0], scale_11);
                    }
                    {
                        v8_2[2] = cake_dsv4_qmul4_portable<5>(kraw_11[1], scale_11);
                    }
                    {
                        v8_2[3] = cake_dsv4_qmul4_portable<6>(kraw_11[1], scale_11);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_2[(0) + 3])));
                int vblock_0_1 = 32 * o_chunk_1 + 12;
                unsigned int v8_1_1[4];
                {
                    int vchunk_12 = vblock_0_1 >> 1;
                    int vhalf_12 = vblock_0_1 & 1;
                    unsigned int kraw_12[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_12[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_12 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_12 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_12)));
                    unsigned int sfw32_12 = smem_sfs32[row_1 * 32 + vblock_0_1 >> 2];
                    unsigned int scale_12 = sfw32_12 >> (unsigned int)(8 * (vblock_0_1 & 3)) & 255;
                    {
                        v8_1_1[0] = cake_dsv4_qmul4_portable<5>(kraw_12[0], scale_12);
                    }
                    {
                        v8_1_1[1] = cake_dsv4_qmul4_portable<6>(kraw_12[0], scale_12);
                    }
                    {
                        v8_1_1[2] = cake_dsv4_qmul4_portable<5>(kraw_12[1], scale_12);
                    }
                    {
                        v8_1_1[3] = cake_dsv4_qmul4_portable<6>(kraw_12[1], scale_12);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_1[(0) + 3])));
                int vblock_2_1 = 32 * o_chunk_1 + 13;
                unsigned int v8_3_1[4];
                {
                    int vchunk_13 = vblock_2_1 >> 1;
                    int vhalf_13 = vblock_2_1 & 1;
                    unsigned int kraw_13[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_13[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_13 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_13 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_13)));
                    unsigned int sfw32_13 = smem_sfs32[row_1 * 32 + vblock_2_1 >> 2];
                    unsigned int scale_13 = sfw32_13 >> (unsigned int)(8 * (vblock_2_1 & 3)) & 255;
                    {
                        v8_3_1[0] = cake_dsv4_qmul4_portable<5>(kraw_13[0], scale_13);
                    }
                    {
                        v8_3_1[1] = cake_dsv4_qmul4_portable<6>(kraw_13[0], scale_13);
                    }
                    {
                        v8_3_1[2] = cake_dsv4_qmul4_portable<5>(kraw_13[1], scale_13);
                    }
                    {
                        v8_3_1[3] = cake_dsv4_qmul4_portable<6>(kraw_13[1], scale_13);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_1[(0) + 3])));
                int vblock_4_1 = 32 * o_chunk_1 + 14;
                unsigned int v8_5_1[4];
                {
                    int vchunk_14 = vblock_4_1 >> 1;
                    int vhalf_14 = vblock_4_1 & 1;
                    unsigned int kraw_14[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_14[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_14 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_14 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_14)));
                    unsigned int sfw32_14 = smem_sfs32[row_1 * 32 + vblock_4_1 >> 2];
                    unsigned int scale_14 = sfw32_14 >> (unsigned int)(8 * (vblock_4_1 & 3)) & 255;
                    {
                        v8_5_1[0] = cake_dsv4_qmul4_portable<5>(kraw_14[0], scale_14);
                    }
                    {
                        v8_5_1[1] = cake_dsv4_qmul4_portable<6>(kraw_14[0], scale_14);
                    }
                    {
                        v8_5_1[2] = cake_dsv4_qmul4_portable<5>(kraw_14[1], scale_14);
                    }
                    {
                        v8_5_1[3] = cake_dsv4_qmul4_portable<6>(kraw_14[1], scale_14);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (96 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_1[(0) + 3])));
                int vblock_6_1 = 32 * o_chunk_1 + 15;
                unsigned int v8_7_1[4];
                {
                    int vchunk_15 = vblock_6_1 >> 1;
                    int vhalf_15 = vblock_6_1 & 1;
                    unsigned int kraw_15[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_15[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_15 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_15 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_15)));
                    unsigned int sfw32_15 = smem_sfs32[row_1 * 32 + vblock_6_1 >> 2];
                    unsigned int scale_15 = sfw32_15 >> (unsigned int)(8 * (vblock_6_1 & 3)) & 255;
                    {
                        v8_7_1[0] = cake_dsv4_qmul4_portable<5>(kraw_15[0], scale_15);
                    }
                    {
                        v8_7_1[1] = cake_dsv4_qmul4_portable<6>(kraw_15[0], scale_15);
                    }
                    {
                        v8_7_1[2] = cake_dsv4_qmul4_portable<5>(kraw_15[1], scale_15);
                    }
                    {
                        v8_7_1[3] = cake_dsv4_qmul4_portable<6>(kraw_15[1], scale_15);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (112 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_1[(0) + 3])));
                int vblock_8_1 = 32 * o_chunk_1 + 16;
                unsigned int v8_9_1[4];
                {
                    int vchunk_16 = vblock_8_1 >> 1;
                    int vhalf_16 = vblock_8_1 & 1;
                    unsigned int kraw_16[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_16[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_16 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_16 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_16)));
                    unsigned int sfw32_16 = smem_sfs32[row_1 * 32 + vblock_8_1 >> 2];
                    unsigned int scale_16 = sfw32_16 >> (unsigned int)(8 * (vblock_8_1 & 3)) & 255;
                    {
                        v8_9_1[0] = cake_dsv4_qmul4_portable<5>(kraw_16[0], scale_16);
                    }
                    {
                        v8_9_1[1] = cake_dsv4_qmul4_portable<6>(kraw_16[0], scale_16);
                    }
                    {
                        v8_9_1[2] = cake_dsv4_qmul4_portable<5>(kraw_16[1], scale_16);
                    }
                    {
                        v8_9_1[3] = cake_dsv4_qmul4_portable<6>(kraw_16[1], scale_16);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (0 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_1[(0) + 3])));
                int vblock_10_1 = 32 * o_chunk_1 + 17;
                unsigned int v8_11_1[4];
                {
                    int vchunk_17 = vblock_10_1 >> 1;
                    int vhalf_17 = vblock_10_1 & 1;
                    unsigned int kraw_17[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_17[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_17 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_17 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_17)));
                    unsigned int sfw32_17 = smem_sfs32[row_1 * 32 + vblock_10_1 >> 2];
                    unsigned int scale_17 = sfw32_17 >> (unsigned int)(8 * (vblock_10_1 & 3)) & 255;
                    {
                        v8_11_1[0] = cake_dsv4_qmul4_portable<5>(kraw_17[0], scale_17);
                    }
                    {
                        v8_11_1[1] = cake_dsv4_qmul4_portable<6>(kraw_17[0], scale_17);
                    }
                    {
                        v8_11_1[2] = cake_dsv4_qmul4_portable<5>(kraw_17[1], scale_17);
                    }
                    {
                        v8_11_1[3] = cake_dsv4_qmul4_portable<6>(kraw_17[1], scale_17);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (16 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_1[(0) + 3])));
                int vblock_12_1 = 32 * o_chunk_1 + 18;
                unsigned int v8_13_1[4];
                {
                    int vchunk_18 = vblock_12_1 >> 1;
                    int vhalf_18 = vblock_12_1 & 1;
                    unsigned int kraw_18[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_18[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_18 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_18 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_18)));
                    unsigned int sfw32_18 = smem_sfs32[row_1 * 32 + vblock_12_1 >> 2];
                    unsigned int scale_18 = sfw32_18 >> (unsigned int)(8 * (vblock_12_1 & 3)) & 255;
                    {
                        v8_13_1[0] = cake_dsv4_qmul4_portable<5>(kraw_18[0], scale_18);
                    }
                    {
                        v8_13_1[1] = cake_dsv4_qmul4_portable<6>(kraw_18[0], scale_18);
                    }
                    {
                        v8_13_1[2] = cake_dsv4_qmul4_portable<5>(kraw_18[1], scale_18);
                    }
                    {
                        v8_13_1[3] = cake_dsv4_qmul4_portable<6>(kraw_18[1], scale_18);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (32 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_1[(0) + 3])));
                int vblock_14_1 = 32 * o_chunk_1 + 19;
                unsigned int v8_15_1[4];
                {
                    int vchunk_19 = vblock_14_1 >> 1;
                    int vhalf_19 = vblock_14_1 & 1;
                    unsigned int kraw_19[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_19[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_19 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_19 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_19)));
                    unsigned int sfw32_19 = smem_sfs32[row_1 * 32 + vblock_14_1 >> 2];
                    unsigned int scale_19 = sfw32_19 >> (unsigned int)(8 * (vblock_14_1 & 3)) & 255;
                    {
                        v8_15_1[0] = cake_dsv4_qmul4_portable<5>(kraw_19[0], scale_19);
                    }
                    {
                        v8_15_1[1] = cake_dsv4_qmul4_portable<6>(kraw_19[0], scale_19);
                    }
                    {
                        v8_15_1[2] = cake_dsv4_qmul4_portable<5>(kraw_19[1], scale_19);
                    }
                    {
                        v8_15_1[3] = cake_dsv4_qmul4_portable<6>(kraw_19[1], scale_19);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_1[(0) + 3])));
                int vblock_16_1 = 32 * o_chunk_1 + 20;
                unsigned int v8_17_1[4];
                {
                    int vchunk_20 = vblock_16_1 >> 1;
                    int vhalf_20 = vblock_16_1 & 1;
                    unsigned int kraw_20[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_20[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_20 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_20 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_20)));
                    unsigned int sfw32_20 = smem_sfs32[row_1 * 32 + vblock_16_1 >> 2];
                    unsigned int scale_20 = sfw32_20 >> (unsigned int)(8 * (vblock_16_1 & 3)) & 255;
                    {
                        v8_17_1[0] = cake_dsv4_qmul4_portable<5>(kraw_20[0], scale_20);
                    }
                    {
                        v8_17_1[1] = cake_dsv4_qmul4_portable<6>(kraw_20[0], scale_20);
                    }
                    {
                        v8_17_1[2] = cake_dsv4_qmul4_portable<5>(kraw_20[1], scale_20);
                    }
                    {
                        v8_17_1[3] = cake_dsv4_qmul4_portable<6>(kraw_20[1], scale_20);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_1[(0) + 3])));
                int vblock_18_1 = 32 * o_chunk_1 + 21;
                unsigned int v8_19_1[4];
                {
                    int vchunk_21 = vblock_18_1 >> 1;
                    int vhalf_21 = vblock_18_1 & 1;
                    unsigned int kraw_21[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_21[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_21 / 8 * 16384) + (unsigned int)(row_1 * 128 + (vchunk_21 % 8 * 16 ^ row_1 % 8 * 16)) + (unsigned int)(8 * vhalf_21)));
                    unsigned int sfw32_21 = smem_sfs32[row_1 * 32 + vblock_18_1 >> 2];
                    unsigned int scale_21 = sfw32_21 >> (unsigned int)(8 * (vblock_18_1 & 3)) & 255;
                    {
                        v8_19_1[0] = cake_dsv4_qmul4_portable<5>(kraw_21[0], scale_21);
                    }
                    {
                        v8_19_1[1] = cake_dsv4_qmul4_portable<6>(kraw_21[0], scale_21);
                    }
                    {
                        v8_19_1[2] = cake_dsv4_qmul4_portable<5>(kraw_21[1], scale_21);
                    }
                    {
                        v8_19_1[3] = cake_dsv4_qmul4_portable<6>(kraw_21[1], scale_21);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_19_1[(0) + 3])));
            } else {
                {
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 128 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 256 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                    smem_ksf32[2048 + row_1 % 32 / 8 * 512 + 384 + row_1 % 8 * 16 + row_1 / 32 % 4 * 4 >> 2] = 0;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (96 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(16384 + (row_1 * 128 + (112 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (0 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (16 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (32 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (48 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (64 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_1 * 128 + (80 ^ row_1 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_1 = bmm2_scale[0];
            unsigned int _phase_s_full_0_1 = 0;
            unsigned int _phase_o_full_0_1 = 0;
            unsigned int _phase_o_full_1_1 = 0;
            unsigned int _phase_o_full_2_1 = 0;
            unsigned int _phase_o_full_3_1 = 0;
            {
                mbarrier_wait_hint(s_full_addr, _phase_s_full_0_1, 10000000);
                _phase_s_full_0_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float score_values_1[4];
                tmem_ld_x4(&score_values_1[0], taddr + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (valid_1 == 0) {
                    score_values_1[0] = -CAKE_INF;
                    score_values_1[1] = -CAKE_INF;
                    score_values_1[2] = -CAKE_INF;
                    score_values_1[3] = -CAKE_INF;
                }
                float tr_vals_1[4];
                tr_vals_1[0] = score_values_1[0];
                tr_vals_1[1] = score_values_1[1];
                tr_vals_1[2] = score_values_1[2];
                tr_vals_1[3] = score_values_1[3];
                int hi_bit_1 = lane & 16;
                float send_1 = ((hi_bit_1 != 0) ? tr_vals_1[0] : tr_vals_1[2]);
                float keep_2 = ((hi_bit_1 != 0) ? tr_vals_1[2] : tr_vals_1[0]);
                float _shfl_43;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_43) : "f"(send_1), "r"(lane ^ 16));
                float recv_1 = _shfl_43;
                float _max_73 = max_noftz(keep_2, recv_1);
                tr_vals_1[0] = _max_73;
                float send_0_1 = ((hi_bit_1 != 0) ? tr_vals_1[1] : tr_vals_1[3]);
                float keep_1_1 = ((hi_bit_1 != 0) ? tr_vals_1[3] : tr_vals_1[1]);
                float _shfl_44;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_44) : "f"(send_0_1), "r"(lane ^ 16));
                float recv_2_1 = _shfl_44;
                float _max_74 = max_noftz(keep_1_1, recv_2_1);
                tr_vals_1[1] = _max_74;
                int hi_bit_3 = lane & 8;
                float send_4 = ((hi_bit_3 != 0) ? tr_vals_1[0] : tr_vals_1[1]);
                float keep_5 = ((hi_bit_3 != 0) ? tr_vals_1[1] : tr_vals_1[0]);
                float _shfl_45;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_45) : "f"(send_4), "r"(lane ^ 8));
                float recv_6 = _shfl_45;
                float _max_75 = max_noftz(keep_5, recv_6);
                tr_vals_1[0] = _max_75;
                float _shfl_46;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_46) : "f"(tr_vals_1[0]), "r"(lane ^ 4));
                float other_1 = _shfl_46;
                float _max_76 = max_noftz(tr_vals_1[0], other_1);
                tr_vals_1[0] = _max_76;
                float _shfl_47;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_47) : "f"(tr_vals_1[0]), "r"(lane ^ 2));
                float other_7 = _shfl_47;
                float _max_77 = max_noftz(tr_vals_1[0], other_7);
                tr_vals_1[0] = _max_77;
                float _shfl_48;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_48) : "f"(tr_vals_1[0]), "r"(lane ^ 1));
                float other_8 = _shfl_48;
                float _max_78 = max_noftz(tr_vals_1[0], other_8);
                tr_vals_1[0] = _max_78;
                if ((lane & 7) == 0) {
                    smem_pmax[(4 + local_warp_1) * 8 + (lane >> 3)] = tr_vals_1[0];
                }
                asm volatile("barrier.sync 11, 128;" ::: "memory");
                float m_lane_1 = -CAKE_INF;
                float sink_lane_1 = -CAKE_INF;
                float m_scaled_1 = 0.0f;
                if (lane < 4) {
                    float _max_79 = max_noftz(smem_pmax[32 + lane], smem_pmax[40 + lane]);
                    float _max_80 = max_noftz(smem_pmax[48 + lane], smem_pmax[56 + lane]);
                    float _max_81 = max_noftz(_max_79, _max_80);
                    m_lane_1 = _max_81;
                    if (has_sinks != 0 && split_idx_1 == 0 && head_base_1 + 8 + lane < num_heads) {
                        sink_lane_1 = sinks[head_base_1 + 8 + lane] * 1.4426950408889634f;
                    }
                    float _max_82 = max_noftz(m_lane_1 * softmax_scale_log2_1, sink_lane_1);
                    m_scaled_1 = _max_82;
                    if (m_scaled_1 == -CAKE_INF) {
                        m_scaled_1 = 0.0f;
                    }
                }
                float col_max_1[4];
                float _shfl_49;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_49) : "f"(m_scaled_1), "r"(0));
                col_max_1[0] = _shfl_49;
                float _shfl_50;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_50) : "f"(m_scaled_1), "r"(1));
                col_max_1[1] = _shfl_50;
                float _shfl_51;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_51) : "f"(m_scaled_1), "r"(2));
                col_max_1[2] = _shfl_51;
                float _shfl_52;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_52) : "f"(m_scaled_1), "r"(3));
                col_max_1[3] = _shfl_52;
                float _exp2_9 = approx_exp2(score_values_1[0] * softmax_scale_log2_1 - col_max_1[0]);
                score_values_1[0] = _exp2_9;
                float _exp2_10 = approx_exp2(score_values_1[1] * softmax_scale_log2_1 - col_max_1[1]);
                score_values_1[1] = _exp2_10;
                float _exp2_11 = approx_exp2(score_values_1[2] * softmax_scale_log2_1 - col_max_1[2]);
                score_values_1[2] = _exp2_11;
                float _exp2_12 = approx_exp2(score_values_1[3] * softmax_scale_log2_1 - col_max_1[3]);
                score_values_1[3] = _exp2_12;
                {
                    uint16_t _fp8_pair_46;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_46) : "f"(0.0f), "f"(score_values_1[0]));
                    uint32_t _byte_46 = (uint32_t)(_fp8_pair_46 & 0xFF);
                    uint32_t _addr_46 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1024 + row_1 ^ (1024 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_46), "r"(_byte_46) : "memory");
                }
                float _fp8_rt_8;
                uint16_t _e4m3x2_47;
                uint32_t _f16x2_47;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_47) : "f"(0.0f), "f"(score_values_1[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_47) : "h"(_e4m3x2_47));
                uint16_t _fp8_h0_47 = (uint16_t)(_f16x2_47 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_8) : "h"(_fp8_h0_47));
                tr_vals_1[0] = _fp8_rt_8;
                {
                    uint16_t _fp8_pair_48;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_48) : "f"(0.0f), "f"(score_values_1[1]));
                    uint32_t _byte_48 = (uint32_t)(_fp8_pair_48 & 0xFF);
                    uint32_t _addr_48 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1152 + row_1 ^ (1152 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_48), "r"(_byte_48) : "memory");
                }
                float _fp8_rt_9;
                uint16_t _e4m3x2_49;
                uint32_t _f16x2_49;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_49) : "f"(0.0f), "f"(score_values_1[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_49) : "h"(_e4m3x2_49));
                uint16_t _fp8_h0_49 = (uint16_t)(_f16x2_49 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_9) : "h"(_fp8_h0_49));
                tr_vals_1[1] = _fp8_rt_9;
                {
                    uint16_t _fp8_pair_50;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_50) : "f"(0.0f), "f"(score_values_1[2]));
                    uint32_t _byte_50 = (uint32_t)(_fp8_pair_50 & 0xFF);
                    uint32_t _addr_50 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1280 + row_1 ^ (1280 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_50), "r"(_byte_50) : "memory");
                }
                float _fp8_rt_10;
                uint16_t _e4m3x2_51;
                uint32_t _f16x2_51;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_51) : "f"(0.0f), "f"(score_values_1[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_51) : "h"(_e4m3x2_51));
                uint16_t _fp8_h0_51 = (uint16_t)(_f16x2_51 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_10) : "h"(_fp8_h0_51));
                tr_vals_1[2] = _fp8_rt_10;
                {
                    uint16_t _fp8_pair_52;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_52) : "f"(0.0f), "f"(score_values_1[3]));
                    uint32_t _byte_52 = (uint32_t)(_fp8_pair_52 & 0xFF);
                    uint32_t _addr_52 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1408 + row_1 ^ (1408 + row_1 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_52), "r"(_byte_52) : "memory");
                }
                float _fp8_rt_11;
                uint16_t _e4m3x2_53;
                uint32_t _f16x2_53;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_53) : "f"(0.0f), "f"(score_values_1[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_53) : "h"(_e4m3x2_53));
                uint16_t _fp8_h0_53 = (uint16_t)(_f16x2_53 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_11) : "h"(_fp8_h0_53));
                tr_vals_1[3] = _fp8_rt_11;
                int hi_bit_9_1 = lane & 16;
                float send_10_1 = ((hi_bit_9_1 != 0) ? score_values_1[0] : score_values_1[2]);
                float keep_11_1 = ((hi_bit_9_1 != 0) ? score_values_1[2] : score_values_1[0]);
                float _shfl_53;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_53) : "f"(send_10_1), "r"(lane ^ 16));
                float recv_12_1 = _shfl_53;
                score_values_1[0] = keep_11_1 + recv_12_1;
                float send_13_1 = ((hi_bit_9_1 != 0) ? score_values_1[1] : score_values_1[3]);
                float keep_14_1 = ((hi_bit_9_1 != 0) ? score_values_1[3] : score_values_1[1]);
                float _shfl_54;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_54) : "f"(send_13_1), "r"(lane ^ 16));
                float recv_15_1 = _shfl_54;
                score_values_1[1] = keep_14_1 + recv_15_1;
                int hi_bit_16_1 = lane & 8;
                float send_17_1 = ((hi_bit_16_1 != 0) ? score_values_1[0] : score_values_1[1]);
                float keep_18_1 = ((hi_bit_16_1 != 0) ? score_values_1[1] : score_values_1[0]);
                float _shfl_55;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_55) : "f"(send_17_1), "r"(lane ^ 8));
                float recv_19_1 = _shfl_55;
                score_values_1[0] = keep_18_1 + recv_19_1;
                float _shfl_56;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_56) : "f"(score_values_1[0]), "r"(lane ^ 4));
                float other_20_1 = _shfl_56;
                score_values_1[0] = score_values_1[0] + other_20_1;
                float _shfl_57;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_57) : "f"(score_values_1[0]), "r"(lane ^ 2));
                float other_21 = _shfl_57;
                score_values_1[0] = score_values_1[0] + other_21;
                float _shfl_58;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_58) : "f"(score_values_1[0]), "r"(lane ^ 1));
                float other_22 = _shfl_58;
                score_values_1[0] = score_values_1[0] + other_22;
                int hi_bit_23 = lane & 16;
                float send_24 = ((hi_bit_23 != 0) ? tr_vals_1[0] : tr_vals_1[2]);
                float keep_25 = ((hi_bit_23 != 0) ? tr_vals_1[2] : tr_vals_1[0]);
                float _shfl_59;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_59) : "f"(send_24), "r"(lane ^ 16));
                float recv_26 = _shfl_59;
                tr_vals_1[0] = keep_25 + recv_26;
                float send_27 = ((hi_bit_23 != 0) ? tr_vals_1[1] : tr_vals_1[3]);
                float keep_28 = ((hi_bit_23 != 0) ? tr_vals_1[3] : tr_vals_1[1]);
                float _shfl_60;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_60) : "f"(send_27), "r"(lane ^ 16));
                float recv_29 = _shfl_60;
                tr_vals_1[1] = keep_28 + recv_29;
                int hi_bit_30 = lane & 8;
                float send_31_1 = ((hi_bit_30 != 0) ? tr_vals_1[0] : tr_vals_1[1]);
                float keep_32_1 = ((hi_bit_30 != 0) ? tr_vals_1[1] : tr_vals_1[0]);
                float _shfl_61;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_61) : "f"(send_31_1), "r"(lane ^ 8));
                float recv_33_1 = _shfl_61;
                tr_vals_1[0] = keep_32_1 + recv_33_1;
                float _shfl_62;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_62) : "f"(tr_vals_1[0]), "r"(lane ^ 4));
                float other_34 = _shfl_62;
                tr_vals_1[0] = tr_vals_1[0] + other_34;
                float _shfl_63;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_63) : "f"(tr_vals_1[0]), "r"(lane ^ 2));
                float other_35 = _shfl_63;
                tr_vals_1[0] = tr_vals_1[0] + other_35;
                float _shfl_64;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_64) : "f"(tr_vals_1[0]), "r"(lane ^ 1));
                float other_36 = _shfl_64;
                tr_vals_1[0] = tr_vals_1[0] + other_36;
                if ((lane & 7) == 0) {
                    smem_psum[(4 + local_warp_1) * 8 + (lane >> 3)] = score_values_1[0];
                    smem_rsum[(4 + local_warp_1) * 8 + (lane >> 3)] = tr_vals_1[0];
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                asm volatile("barrier.sync 11, 128;" ::: "memory");
                float norm_lane_1 = 0.0f;
                if (lane < 4) {
                    float _exp2_13 = approx_exp2(sink_lane_1 - m_scaled_1);
                    float sink_term_1 = _exp2_13;
                    float col_sum_1 = smem_psum[32 + lane] + smem_psum[40 + lane] + smem_psum[48 + lane] + smem_psum[56 + lane] + sink_term_1;
                    float denom_1 = smem_rsum[32 + lane] + smem_rsum[40 + lane] + smem_rsum[48 + lane] + smem_rsum[56 + lane] + sink_term_1;
                    if (denom_1 > 0.0f) {
                        float _rcp_1 = approx_rcp(denom_1);
                        norm_lane_1 = _rcp_1 * output_scale_1;
                    }
                    if (local_warp_1 == 0) {
                        if (o_chunk_1 == 0 && head_base_1 + 8 + lane < num_heads) {
                            int lse_offset_1 = (query_idx_1 * num_heads + head_base_1 + 8 + lane) * num_splits + split_idx_1;
                            float _log2_1;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(col_sum_1));
                            partial_lse[lse_offset_1] = ((col_sum_1 > 0.0f) ? (m_scaled_1 + _log2_1) * lse_partial_scale : -CAKE_INF);
                        }
                    }
                }
                float norm_c_1[4];
                float _shfl_65;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_65) : "f"(norm_lane_1), "r"(0));
                norm_c_1[0] = _shfl_65;
                float _shfl_66;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_66) : "f"(norm_lane_1), "r"(1));
                norm_c_1[1] = _shfl_66;
                float _shfl_67;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_67) : "f"(norm_lane_1), "r"(2));
                norm_c_1[2] = _shfl_67;
                float _shfl_68;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_68) : "f"(norm_lane_1), "r"(3));
                norm_c_1[3] = _shfl_68;
                float o_values_1[4];
                int dim_1 = 0;
                long long out_off_1 = 0;
                mbarrier_wait_hint(o_full_addr, _phase_o_full_0_1, 10000000);
                _phase_o_full_0_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x4(&o_values_1[0], taddr + 16 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_1 = o_chunk_1 * 4 * 128 + row_1;
                if (head_base_1 + 8 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[0] * norm_c_1[0];
                }
                if (head_base_1 + 8 + 1 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 1) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[1] * norm_c_1[1];
                }
                if (head_base_1 + 8 + 2 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 2) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[2] * norm_c_1[2];
                }
                if (head_base_1 + 8 + 3 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 3) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[3] * norm_c_1[3];
                }
                mbarrier_wait_hint(o_full_addr + 8, _phase_o_full_1_1, 10000000);
                _phase_o_full_1_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x4(&o_values_1[0], taddr + 32 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_1 = (o_chunk_1 * 4 + 1) * 128 + row_1;
                if (head_base_1 + 8 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[0] * norm_c_1[0];
                }
                if (head_base_1 + 8 + 1 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 1) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[1] * norm_c_1[1];
                }
                if (head_base_1 + 8 + 2 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 2) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[2] * norm_c_1[2];
                }
                if (head_base_1 + 8 + 3 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 3) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[3] * norm_c_1[3];
                }
                mbarrier_wait_hint(o_full_addr + 16, _phase_o_full_2_1, 10000000);
                _phase_o_full_2_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x4(&o_values_1[0], taddr + 48 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_1 = (o_chunk_1 * 4 + 2) * 128 + row_1;
                if (head_base_1 + 8 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[0] * norm_c_1[0];
                }
                if (head_base_1 + 8 + 1 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 1) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[1] * norm_c_1[1];
                }
                if (head_base_1 + 8 + 2 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 2) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[2] * norm_c_1[2];
                }
                if (head_base_1 + 8 + 3 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 3) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[3] * norm_c_1[3];
                }
                mbarrier_wait_hint(o_full_addr + 24, _phase_o_full_3_1, 10000000);
                _phase_o_full_3_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x4(&o_values_1[0], taddr + 64 + 8 + (unsigned int)(tmem_row_origin_1 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_1 = (o_chunk_1 * 4 + 3) * 128 + row_1;
                if (head_base_1 + 8 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[0] * norm_c_1[0];
                }
                if (head_base_1 + 8 + 1 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 1) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[1] * norm_c_1[1];
                }
                if (head_base_1 + 8 + 2 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 2) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[2] * norm_c_1[2];
                }
                if (head_base_1 + 8 + 3 < num_heads) {
                    out_off_1 = ((long long)(query_idx_1 * num_heads + head_base_1 + 8 + 3) * (long long)num_splits + (long long)split_idx_1) * 512 + (long long)dim_1;
                    partial_O[out_off_1] = o_values_1[3] * norm_c_1[3];
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
            int o_chunk_2 = 0;
            int work_idx_2 = blockIdx.x;
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
                    if (head_base_2 + q_block_2 * 32 < num_heads && q_row_2 < 16) {
                        int q_row_addr_2 = smem_qstage_addr + (unsigned int)(kset_2 * 2048) + (unsigned int)(q_row_2 * 128);
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
                            float _max_83 = max_noftz(_fabs_64, _fabs_65);
                            m8_2[0] = _max_83;
                            float _fabs_66 = fabsf(qv_3[2]);
                            float _fabs_67 = fabsf(qv_3[3]);
                            float _max_84 = max_noftz(_fabs_66, _fabs_67);
                            m8_2[1] = _max_84;
                            float _fabs_68 = fabsf(qv_3[4]);
                            float _fabs_69 = fabsf(qv_3[5]);
                            float _max_85 = max_noftz(_fabs_68, _fabs_69);
                            m8_2[2] = _max_85;
                            float _fabs_70 = fabsf(qv_3[6]);
                            float _fabs_71 = fabsf(qv_3[7]);
                            float _max_86 = max_noftz(_fabs_70, _fabs_71);
                            m8_2[3] = _max_86;
                            float _fabs_72 = fabsf(qv_3[8]);
                            float _fabs_73 = fabsf(qv_3[9]);
                            float _max_87 = max_noftz(_fabs_72, _fabs_73);
                            m8_2[4] = _max_87;
                            float _fabs_74 = fabsf(qv_3[10]);
                            float _fabs_75 = fabsf(qv_3[11]);
                            float _max_88 = max_noftz(_fabs_74, _fabs_75);
                            m8_2[5] = _max_88;
                            float _fabs_76 = fabsf(qv_3[12]);
                            float _fabs_77 = fabsf(qv_3[13]);
                            float _max_89 = max_noftz(_fabs_76, _fabs_77);
                            m8_2[6] = _max_89;
                            float _fabs_78 = fabsf(qv_3[14]);
                            float _fabs_79 = fabsf(qv_3[15]);
                            float _max_90 = max_noftz(_fabs_78, _fabs_79);
                            m8_2[7] = _max_90;
                            float m4_2[4];
                            float _max_91 = max_noftz(m8_2[0], m8_2[1]);
                            m4_2[0] = _max_91;
                            float _max_92 = max_noftz(m8_2[2], m8_2[3]);
                            m4_2[1] = _max_92;
                            float _max_93 = max_noftz(m8_2[4], m8_2[5]);
                            m4_2[2] = _max_93;
                            float _max_94 = max_noftz(m8_2[6], m8_2[7]);
                            m4_2[3] = _max_94;
                            float _max_95 = max_noftz(m4_2[0], m4_2[1]);
                            float _max_96 = max_noftz(m4_2[2], m4_2[3]);
                            float _max_97 = max_noftz(_max_95, _max_96);
                            float amax_2 = _max_97;
                            float sc_2 = amax_2 * inv_six_2;
                            uint16_t _e4m3x2_f32_356;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_356) : "f"(0.0f), "f"(sc_2));
                            uint16_t sc_pair_2 = _e4m3x2_f32_356;
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
                            float _max_98 = max_noftz(_fabs_80, _fabs_81);
                            m8_3_2[0] = _max_98;
                            float _fabs_82 = fabsf(qv_2_2[2]);
                            float _fabs_83 = fabsf(qv_2_2[3]);
                            float _max_99 = max_noftz(_fabs_82, _fabs_83);
                            m8_3_2[1] = _max_99;
                            float _fabs_84 = fabsf(qv_2_2[4]);
                            float _fabs_85 = fabsf(qv_2_2[5]);
                            float _max_100 = max_noftz(_fabs_84, _fabs_85);
                            m8_3_2[2] = _max_100;
                            float _fabs_86 = fabsf(qv_2_2[6]);
                            float _fabs_87 = fabsf(qv_2_2[7]);
                            float _max_101 = max_noftz(_fabs_86, _fabs_87);
                            m8_3_2[3] = _max_101;
                            float _fabs_88 = fabsf(qv_2_2[8]);
                            float _fabs_89 = fabsf(qv_2_2[9]);
                            float _max_102 = max_noftz(_fabs_88, _fabs_89);
                            m8_3_2[4] = _max_102;
                            float _fabs_90 = fabsf(qv_2_2[10]);
                            float _fabs_91 = fabsf(qv_2_2[11]);
                            float _max_103 = max_noftz(_fabs_90, _fabs_91);
                            m8_3_2[5] = _max_103;
                            float _fabs_92 = fabsf(qv_2_2[12]);
                            float _fabs_93 = fabsf(qv_2_2[13]);
                            float _max_104 = max_noftz(_fabs_92, _fabs_93);
                            m8_3_2[6] = _max_104;
                            float _fabs_94 = fabsf(qv_2_2[14]);
                            float _fabs_95 = fabsf(qv_2_2[15]);
                            float _max_105 = max_noftz(_fabs_94, _fabs_95);
                            m8_3_2[7] = _max_105;
                            float m4_4_2[4];
                            float _max_106 = max_noftz(m8_3_2[0], m8_3_2[1]);
                            m4_4_2[0] = _max_106;
                            float _max_107 = max_noftz(m8_3_2[2], m8_3_2[3]);
                            m4_4_2[1] = _max_107;
                            float _max_108 = max_noftz(m8_3_2[4], m8_3_2[5]);
                            m4_4_2[2] = _max_108;
                            float _max_109 = max_noftz(m8_3_2[6], m8_3_2[7]);
                            m4_4_2[3] = _max_109;
                            float _max_110 = max_noftz(m4_4_2[0], m4_4_2[1]);
                            float _max_111 = max_noftz(m4_4_2[2], m4_4_2[3]);
                            float _max_112 = max_noftz(_max_110, _max_111);
                            float amax_5_2 = _max_112;
                            float sc_6_2 = amax_5_2 * inv_six_2;
                            uint16_t _e4m3x2_f32_357;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_357) : "f"(0.0f), "f"(sc_6_2));
                            uint16_t sc_pair_7_2 = _e4m3x2_f32_357;
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
                int vblock_3 = 32 * o_chunk_2 + 22;
                unsigned int v8_4[4];
                {
                    int vchunk_22 = vblock_3 >> 1;
                    int vhalf_22 = vblock_3 & 1;
                    unsigned int kraw_22[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_22[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_22 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_22 % 8 * 16 ^ row_2 % 8 * 16)) + (unsigned int)(8 * vhalf_22)));
                    unsigned int sfw32_22 = smem_sfs32[row_2 * 32 + vblock_3 >> 2];
                    unsigned int scale_22 = sfw32_22 >> (unsigned int)(8 * (vblock_3 & 3)) & 255;
                    {
                        v8_4[0] = cake_dsv4_qmul4_portable<5>(kraw_22[0], scale_22);
                    }
                    {
                        v8_4[1] = cake_dsv4_qmul4_portable<6>(kraw_22[0], scale_22);
                    }
                    {
                        v8_4[2] = cake_dsv4_qmul4_portable<5>(kraw_22[1], scale_22);
                    }
                    {
                        v8_4[3] = cake_dsv4_qmul4_portable<6>(kraw_22[1], scale_22);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_4[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_4[(0) + 3])));
                int vblock_0_2 = 32 * o_chunk_2 + 23;
                unsigned int v8_1_2[4];
                {
                    int vchunk_23 = vblock_0_2 >> 1;
                    int vhalf_23 = vblock_0_2 & 1;
                    unsigned int kraw_23[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_23[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_23 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_23 % 8 * 16 ^ row_2 % 8 * 16)) + (unsigned int)(8 * vhalf_23)));
                    unsigned int sfw32_23 = smem_sfs32[row_2 * 32 + vblock_0_2 >> 2];
                    unsigned int scale_23 = sfw32_23 >> (unsigned int)(8 * (vblock_0_2 & 3)) & 255;
                    {
                        v8_1_2[0] = cake_dsv4_qmul4_portable<5>(kraw_23[0], scale_23);
                    }
                    {
                        v8_1_2[1] = cake_dsv4_qmul4_portable<6>(kraw_23[0], scale_23);
                    }
                    {
                        v8_1_2[2] = cake_dsv4_qmul4_portable<5>(kraw_23[1], scale_23);
                    }
                    {
                        v8_1_2[3] = cake_dsv4_qmul4_portable<6>(kraw_23[1], scale_23);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_1_2[(0) + 3])));
                int vblock_2_2 = 32 * o_chunk_2 + 24;
                unsigned int v8_3_2[4];
                {
                    int vchunk_24 = vblock_2_2 >> 1;
                    int vhalf_24 = vblock_2_2 & 1;
                    unsigned int kraw_24[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_24[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_24 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_24 % 8 * 16 ^ row_2 % 8 * 16)) + (unsigned int)(8 * vhalf_24)));
                    unsigned int sfw32_24 = smem_sfs32[row_2 * 32 + vblock_2_2 >> 2];
                    unsigned int scale_24 = sfw32_24 >> (unsigned int)(8 * (vblock_2_2 & 3)) & 255;
                    {
                        v8_3_2[0] = cake_dsv4_qmul4_portable<5>(kraw_24[0], scale_24);
                    }
                    {
                        v8_3_2[1] = cake_dsv4_qmul4_portable<6>(kraw_24[0], scale_24);
                    }
                    {
                        v8_3_2[2] = cake_dsv4_qmul4_portable<5>(kraw_24[1], scale_24);
                    }
                    {
                        v8_3_2[3] = cake_dsv4_qmul4_portable<6>(kraw_24[1], scale_24);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (0 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_3_2[(0) + 3])));
                int vblock_4_2 = 32 * o_chunk_2 + 25;
                unsigned int v8_5_2[4];
                {
                    int vchunk_25 = vblock_4_2 >> 1;
                    int vhalf_25 = vblock_4_2 & 1;
                    unsigned int kraw_25[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_25[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_25 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_25 % 8 * 16 ^ row_2 % 8 * 16)) + (unsigned int)(8 * vhalf_25)));
                    unsigned int sfw32_25 = smem_sfs32[row_2 * 32 + vblock_4_2 >> 2];
                    unsigned int scale_25 = sfw32_25 >> (unsigned int)(8 * (vblock_4_2 & 3)) & 255;
                    {
                        v8_5_2[0] = cake_dsv4_qmul4_portable<5>(kraw_25[0], scale_25);
                    }
                    {
                        v8_5_2[1] = cake_dsv4_qmul4_portable<6>(kraw_25[0], scale_25);
                    }
                    {
                        v8_5_2[2] = cake_dsv4_qmul4_portable<5>(kraw_25[1], scale_25);
                    }
                    {
                        v8_5_2[3] = cake_dsv4_qmul4_portable<6>(kraw_25[1], scale_25);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (16 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_5_2[(0) + 3])));
                int vblock_6_2 = 32 * o_chunk_2 + 26;
                unsigned int v8_7_2[4];
                {
                    int vchunk_26 = vblock_6_2 >> 1;
                    int vhalf_26 = vblock_6_2 & 1;
                    unsigned int kraw_26[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_26[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_26 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_26 % 8 * 16 ^ row_2 % 8 * 16)) + (unsigned int)(8 * vhalf_26)));
                    unsigned int sfw32_26 = smem_sfs32[row_2 * 32 + vblock_6_2 >> 2];
                    unsigned int scale_26 = sfw32_26 >> (unsigned int)(8 * (vblock_6_2 & 3)) & 255;
                    {
                        v8_7_2[0] = cake_dsv4_qmul4_portable<5>(kraw_26[0], scale_26);
                    }
                    {
                        v8_7_2[1] = cake_dsv4_qmul4_portable<6>(kraw_26[0], scale_26);
                    }
                    {
                        v8_7_2[2] = cake_dsv4_qmul4_portable<5>(kraw_26[1], scale_26);
                    }
                    {
                        v8_7_2[3] = cake_dsv4_qmul4_portable<6>(kraw_26[1], scale_26);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (32 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_7_2[(0) + 3])));
                int vblock_8_2 = 32 * o_chunk_2 + 27;
                unsigned int v8_9_2[4];
                {
                    int vchunk_27 = vblock_8_2 >> 1;
                    int vhalf_27 = vblock_8_2 & 1;
                    unsigned int kraw_27[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&kraw_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&kraw_27[(0) + 1]))
                        : "r"(smem_kf4_addr + (unsigned int)(vchunk_27 / 8 * 16384) + (unsigned int)(row_2 * 128 + (vchunk_27 % 8 * 16 ^ row_2 % 8 * 16)) + (unsigned int)(8 * vhalf_27)));
                    unsigned int sfw32_27 = smem_sfs32[row_2 * 32 + vblock_8_2 >> 2];
                    unsigned int scale_27 = sfw32_27 >> (unsigned int)(8 * (vblock_8_2 & 3)) & 255;
                    {
                        v8_9_2[0] = cake_dsv4_qmul4_portable<5>(kraw_27[0], scale_27);
                    }
                    {
                        v8_9_2[1] = cake_dsv4_qmul4_portable<6>(kraw_27[0], scale_27);
                    }
                    {
                        v8_9_2[2] = cake_dsv4_qmul4_portable<5>(kraw_27[1], scale_27);
                    }
                    {
                        v8_9_2[3] = cake_dsv4_qmul4_portable<6>(kraw_27[1], scale_27);
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (48 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_9_2[(0) + 3])));
                int vblock_10_2 = 32 * o_chunk_2 + 28;
                unsigned int v8_11_2[4];
                {
                    int rblock = vblock_10_2 - 28;
                    unsigned int rope[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + (2 * rblock * 16 ^ row_2 % 8 * 16))));
                    float lo = __uint_as_float(rope[0] << 16);
                    float hi = __uint_as_float(rope[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_454;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_454) : "f"(hi), "f"(lo));
                    uint16_t pair = _e4m3x2_f32_454;
                    {
                        v8_11_2[0] = (unsigned int)pair;
                    }
                    float lo_0 = __uint_as_float(rope[1] << 16);
                    float hi_1 = __uint_as_float(rope[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_455;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_455) : "f"(hi_1), "f"(lo_0));
                    uint16_t pair_2 = _e4m3x2_f32_455;
                    {
                        v8_11_2[0] = v8_11_2[0] | (unsigned int)pair_2 << 16;
                    }
                    float lo_3 = __uint_as_float(rope[2] << 16);
                    float hi_4 = __uint_as_float(rope[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_456;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_456) : "f"(hi_4), "f"(lo_3));
                    uint16_t pair_5 = _e4m3x2_f32_456;
                    {
                        v8_11_2[1] = (unsigned int)pair_5;
                    }
                    float lo_6 = __uint_as_float(rope[3] << 16);
                    float hi_7 = __uint_as_float(rope[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_457;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_457) : "f"(hi_7), "f"(lo_6));
                    uint16_t pair_8 = _e4m3x2_f32_457;
                    {
                        v8_11_2[1] = v8_11_2[1] | (unsigned int)pair_8 << 16;
                    }
                    unsigned int rope_9[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + ((2 * rblock + 1) * 16 ^ row_2 % 8 * 16))));
                    float lo_10 = __uint_as_float(rope_9[0] << 16);
                    float hi_11 = __uint_as_float(rope_9[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_458;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_458) : "f"(hi_11), "f"(lo_10));
                    uint16_t pair_12 = _e4m3x2_f32_458;
                    {
                        v8_11_2[2] = (unsigned int)pair_12;
                    }
                    float lo_13 = __uint_as_float(rope_9[1] << 16);
                    float hi_14 = __uint_as_float(rope_9[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_459;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_459) : "f"(hi_14), "f"(lo_13));
                    uint16_t pair_15 = _e4m3x2_f32_459;
                    {
                        v8_11_2[2] = v8_11_2[2] | (unsigned int)pair_15 << 16;
                    }
                    float lo_16 = __uint_as_float(rope_9[2] << 16);
                    float hi_17 = __uint_as_float(rope_9[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_460;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_460) : "f"(hi_17), "f"(lo_16));
                    uint16_t pair_18 = _e4m3x2_f32_460;
                    {
                        v8_11_2[3] = (unsigned int)pair_18;
                    }
                    float lo_19 = __uint_as_float(rope_9[3] << 16);
                    float hi_20 = __uint_as_float(rope_9[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_461;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_461) : "f"(hi_20), "f"(lo_19));
                    uint16_t pair_21 = _e4m3x2_f32_461;
                    {
                        v8_11_2[3] = v8_11_2[3] | (unsigned int)pair_21 << 16;
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (64 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_11_2[(0) + 3])));
                int vblock_12_2 = 32 * o_chunk_2 + 29;
                unsigned int v8_13_2[4];
                {
                    int rblock_1 = vblock_12_2 - 28;
                    unsigned int rope_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_1[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + (2 * rblock_1 * 16 ^ row_2 % 8 * 16))));
                    float lo_1 = __uint_as_float(rope_1[0] << 16);
                    float hi_2 = __uint_as_float(rope_1[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_470;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_470) : "f"(hi_2), "f"(lo_1));
                    uint16_t pair_1 = _e4m3x2_f32_470;
                    {
                        v8_13_2[0] = (unsigned int)pair_1;
                    }
                    float lo_0_1 = __uint_as_float(rope_1[1] << 16);
                    float hi_1_1 = __uint_as_float(rope_1[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_471;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_471) : "f"(hi_1_1), "f"(lo_0_1));
                    uint16_t pair_2_1 = _e4m3x2_f32_471;
                    {
                        v8_13_2[0] = v8_13_2[0] | (unsigned int)pair_2_1 << 16;
                    }
                    float lo_3_1 = __uint_as_float(rope_1[2] << 16);
                    float hi_4_1 = __uint_as_float(rope_1[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_472;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_472) : "f"(hi_4_1), "f"(lo_3_1));
                    uint16_t pair_5_1 = _e4m3x2_f32_472;
                    {
                        v8_13_2[1] = (unsigned int)pair_5_1;
                    }
                    float lo_6_1 = __uint_as_float(rope_1[3] << 16);
                    float hi_7_1 = __uint_as_float(rope_1[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_473;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_473) : "f"(hi_7_1), "f"(lo_6_1));
                    uint16_t pair_8_1 = _e4m3x2_f32_473;
                    {
                        v8_13_2[1] = v8_13_2[1] | (unsigned int)pair_8_1 << 16;
                    }
                    unsigned int rope_9_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_1[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_1 + 1) * 16 ^ row_2 % 8 * 16))));
                    float lo_10_1 = __uint_as_float(rope_9_1[0] << 16);
                    float hi_11_1 = __uint_as_float(rope_9_1[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_474;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_474) : "f"(hi_11_1), "f"(lo_10_1));
                    uint16_t pair_12_1 = _e4m3x2_f32_474;
                    {
                        v8_13_2[2] = (unsigned int)pair_12_1;
                    }
                    float lo_13_1 = __uint_as_float(rope_9_1[1] << 16);
                    float hi_14_1 = __uint_as_float(rope_9_1[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_475;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_475) : "f"(hi_14_1), "f"(lo_13_1));
                    uint16_t pair_15_1 = _e4m3x2_f32_475;
                    {
                        v8_13_2[2] = v8_13_2[2] | (unsigned int)pair_15_1 << 16;
                    }
                    float lo_16_1 = __uint_as_float(rope_9_1[2] << 16);
                    float hi_17_1 = __uint_as_float(rope_9_1[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_476;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_476) : "f"(hi_17_1), "f"(lo_16_1));
                    uint16_t pair_18_1 = _e4m3x2_f32_476;
                    {
                        v8_13_2[3] = (unsigned int)pair_18_1;
                    }
                    float lo_19_1 = __uint_as_float(rope_9_1[3] << 16);
                    float hi_20_1 = __uint_as_float(rope_9_1[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_477;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_477) : "f"(hi_20_1), "f"(lo_19_1));
                    uint16_t pair_21_1 = _e4m3x2_f32_477;
                    {
                        v8_13_2[3] = v8_13_2[3] | (unsigned int)pair_21_1 << 16;
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (80 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_13_2[(0) + 3])));
                int vblock_14_2 = 32 * o_chunk_2 + 30;
                unsigned int v8_15_2[4];
                {
                    int rblock_2 = vblock_14_2 - 28;
                    unsigned int rope_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_2[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + (2 * rblock_2 * 16 ^ row_2 % 8 * 16))));
                    float lo_2 = __uint_as_float(rope_2[0] << 16);
                    float hi_3 = __uint_as_float(rope_2[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_486;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_486) : "f"(hi_3), "f"(lo_2));
                    uint16_t pair_3 = _e4m3x2_f32_486;
                    {
                        v8_15_2[0] = (unsigned int)pair_3;
                    }
                    float lo_0_2 = __uint_as_float(rope_2[1] << 16);
                    float hi_1_2 = __uint_as_float(rope_2[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_487;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_487) : "f"(hi_1_2), "f"(lo_0_2));
                    uint16_t pair_2_2 = _e4m3x2_f32_487;
                    {
                        v8_15_2[0] = v8_15_2[0] | (unsigned int)pair_2_2 << 16;
                    }
                    float lo_3_2 = __uint_as_float(rope_2[2] << 16);
                    float hi_4_2 = __uint_as_float(rope_2[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_488;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_488) : "f"(hi_4_2), "f"(lo_3_2));
                    uint16_t pair_5_2 = _e4m3x2_f32_488;
                    {
                        v8_15_2[1] = (unsigned int)pair_5_2;
                    }
                    float lo_6_2 = __uint_as_float(rope_2[3] << 16);
                    float hi_7_2 = __uint_as_float(rope_2[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_489;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_489) : "f"(hi_7_2), "f"(lo_6_2));
                    uint16_t pair_8_2 = _e4m3x2_f32_489;
                    {
                        v8_15_2[1] = v8_15_2[1] | (unsigned int)pair_8_2 << 16;
                    }
                    unsigned int rope_9_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_2[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_2 + 1) * 16 ^ row_2 % 8 * 16))));
                    float lo_10_2 = __uint_as_float(rope_9_2[0] << 16);
                    float hi_11_2 = __uint_as_float(rope_9_2[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_490;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_490) : "f"(hi_11_2), "f"(lo_10_2));
                    uint16_t pair_12_2 = _e4m3x2_f32_490;
                    {
                        v8_15_2[2] = (unsigned int)pair_12_2;
                    }
                    float lo_13_2 = __uint_as_float(rope_9_2[1] << 16);
                    float hi_14_2 = __uint_as_float(rope_9_2[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_491;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_491) : "f"(hi_14_2), "f"(lo_13_2));
                    uint16_t pair_15_2 = _e4m3x2_f32_491;
                    {
                        v8_15_2[2] = v8_15_2[2] | (unsigned int)pair_15_2 << 16;
                    }
                    float lo_16_2 = __uint_as_float(rope_9_2[2] << 16);
                    float hi_17_2 = __uint_as_float(rope_9_2[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_492;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_492) : "f"(hi_17_2), "f"(lo_16_2));
                    uint16_t pair_18_2 = _e4m3x2_f32_492;
                    {
                        v8_15_2[3] = (unsigned int)pair_18_2;
                    }
                    float lo_19_2 = __uint_as_float(rope_9_2[3] << 16);
                    float hi_20_2 = __uint_as_float(rope_9_2[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_493;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_493) : "f"(hi_20_2), "f"(lo_19_2));
                    uint16_t pair_21_2 = _e4m3x2_f32_493;
                    {
                        v8_15_2[3] = v8_15_2[3] | (unsigned int)pair_21_2 << 16;
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_15_2[(0) + 3])));
                int vblock_16_2 = 32 * o_chunk_2 + 31;
                unsigned int v8_17_2[4];
                {
                    int rblock_3 = vblock_16_2 - 28;
                    unsigned int rope_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_3[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + (2 * rblock_3 * 16 ^ row_2 % 8 * 16))));
                    float lo_4 = __uint_as_float(rope_3[0] << 16);
                    float hi_5 = __uint_as_float(rope_3[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_502;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_502) : "f"(hi_5), "f"(lo_4));
                    uint16_t pair_4 = _e4m3x2_f32_502;
                    {
                        v8_17_2[0] = (unsigned int)pair_4;
                    }
                    float lo_0_3 = __uint_as_float(rope_3[1] << 16);
                    float hi_1_3 = __uint_as_float(rope_3[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_503;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_503) : "f"(hi_1_3), "f"(lo_0_3));
                    uint16_t pair_2_3 = _e4m3x2_f32_503;
                    {
                        v8_17_2[0] = v8_17_2[0] | (unsigned int)pair_2_3 << 16;
                    }
                    float lo_3_3 = __uint_as_float(rope_3[2] << 16);
                    float hi_4_3 = __uint_as_float(rope_3[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_504;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_504) : "f"(hi_4_3), "f"(lo_3_3));
                    uint16_t pair_5_3 = _e4m3x2_f32_504;
                    {
                        v8_17_2[1] = (unsigned int)pair_5_3;
                    }
                    float lo_6_3 = __uint_as_float(rope_3[3] << 16);
                    float hi_7_3 = __uint_as_float(rope_3[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_505;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_505) : "f"(hi_7_3), "f"(lo_6_3));
                    uint16_t pair_8_3 = _e4m3x2_f32_505;
                    {
                        v8_17_2[1] = v8_17_2[1] | (unsigned int)pair_8_3 << 16;
                    }
                    unsigned int rope_9_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rope_9_3[(0) + 3]))
                        : "r"(smem_krope_addr + (unsigned int)(row_2 * 128 + ((2 * rblock_3 + 1) * 16 ^ row_2 % 8 * 16))));
                    float lo_10_3 = __uint_as_float(rope_9_3[0] << 16);
                    float hi_11_3 = __uint_as_float(rope_9_3[0] & 4294901760u);
                    uint16_t _e4m3x2_f32_506;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_506) : "f"(hi_11_3), "f"(lo_10_3));
                    uint16_t pair_12_3 = _e4m3x2_f32_506;
                    {
                        v8_17_2[2] = (unsigned int)pair_12_3;
                    }
                    float lo_13_3 = __uint_as_float(rope_9_3[1] << 16);
                    float hi_14_3 = __uint_as_float(rope_9_3[1] & 4294901760u);
                    uint16_t _e4m3x2_f32_507;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_507) : "f"(hi_14_3), "f"(lo_13_3));
                    uint16_t pair_15_3 = _e4m3x2_f32_507;
                    {
                        v8_17_2[2] = v8_17_2[2] | (unsigned int)pair_15_3 << 16;
                    }
                    float lo_16_3 = __uint_as_float(rope_9_3[2] << 16);
                    float hi_17_3 = __uint_as_float(rope_9_3[2] & 4294901760u);
                    uint16_t _e4m3x2_f32_508;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_508) : "f"(hi_17_3), "f"(lo_16_3));
                    uint16_t pair_18_3 = _e4m3x2_f32_508;
                    {
                        v8_17_2[3] = (unsigned int)pair_18_3;
                    }
                    float lo_19_3 = __uint_as_float(rope_9_3[3] << 16);
                    float hi_20_3 = __uint_as_float(rope_9_3[3] & 4294901760u);
                    uint16_t _e4m3x2_f32_509;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_509) : "f"(hi_20_3), "f"(lo_19_3));
                    uint16_t pair_21_3 = _e4m3x2_f32_509;
                    {
                        v8_17_2[3] = v8_17_2[3] | (unsigned int)pair_21_3 << 16;
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[0])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&v8_17_2[(0) + 3])));
            } else {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(32768 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (0 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (16 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (32 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (48 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (64 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (80 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (96 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v_addr + (unsigned int)(49152 + (row_2 * 128 + (112 ^ row_2 % 8 * 16)))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(kv_full_addr);
            float softmax_scale_log2_2 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale_2 = bmm2_scale[0];
            unsigned int _phase_s_full_0_2 = 0;
            unsigned int _phase_o_full_0_2 = 0;
            unsigned int _phase_o_full_1_2 = 0;
            unsigned int _phase_o_full_2_2 = 0;
            unsigned int _phase_o_full_3_2 = 0;
            {
                mbarrier_wait_hint(s_full_addr, _phase_s_full_0_2, 10000000);
                _phase_s_full_0_2 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float score_values_2[4];
                tmem_ld_x4(&score_values_2[0], taddr + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (valid_2 == 0) {
                    score_values_2[0] = -CAKE_INF;
                    score_values_2[1] = -CAKE_INF;
                    score_values_2[2] = -CAKE_INF;
                    score_values_2[3] = -CAKE_INF;
                }
                float tr_vals_2[4];
                tr_vals_2[0] = score_values_2[0];
                tr_vals_2[1] = score_values_2[1];
                tr_vals_2[2] = score_values_2[2];
                tr_vals_2[3] = score_values_2[3];
                int hi_bit_2 = lane & 16;
                float send_2 = ((hi_bit_2 != 0) ? tr_vals_2[0] : tr_vals_2[2]);
                float keep_3 = ((hi_bit_2 != 0) ? tr_vals_2[2] : tr_vals_2[0]);
                float _shfl_69;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_69) : "f"(send_2), "r"(lane ^ 16));
                float recv_3 = _shfl_69;
                float _max_113 = max_noftz(keep_3, recv_3);
                tr_vals_2[0] = _max_113;
                float send_0_2 = ((hi_bit_2 != 0) ? tr_vals_2[1] : tr_vals_2[3]);
                float keep_1_2 = ((hi_bit_2 != 0) ? tr_vals_2[3] : tr_vals_2[1]);
                float _shfl_70;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_70) : "f"(send_0_2), "r"(lane ^ 16));
                float recv_2_2 = _shfl_70;
                float _max_114 = max_noftz(keep_1_2, recv_2_2);
                tr_vals_2[1] = _max_114;
                int hi_bit_3_1 = lane & 8;
                float send_4_1 = ((hi_bit_3_1 != 0) ? tr_vals_2[0] : tr_vals_2[1]);
                float keep_5_1 = ((hi_bit_3_1 != 0) ? tr_vals_2[1] : tr_vals_2[0]);
                float _shfl_71;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_71) : "f"(send_4_1), "r"(lane ^ 8));
                float recv_6_1 = _shfl_71;
                float _max_115 = max_noftz(keep_5_1, recv_6_1);
                tr_vals_2[0] = _max_115;
                float _shfl_72;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_72) : "f"(tr_vals_2[0]), "r"(lane ^ 4));
                float other_2 = _shfl_72;
                float _max_116 = max_noftz(tr_vals_2[0], other_2);
                tr_vals_2[0] = _max_116;
                float _shfl_73;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_73) : "f"(tr_vals_2[0]), "r"(lane ^ 2));
                float other_7_1 = _shfl_73;
                float _max_117 = max_noftz(tr_vals_2[0], other_7_1);
                tr_vals_2[0] = _max_117;
                float _shfl_74;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_74) : "f"(tr_vals_2[0]), "r"(lane ^ 1));
                float other_8_1 = _shfl_74;
                float _max_118 = max_noftz(tr_vals_2[0], other_8_1);
                tr_vals_2[0] = _max_118;
                if ((lane & 7) == 0) {
                    smem_pmax[(8 + local_warp_2) * 8 + (lane >> 3)] = tr_vals_2[0];
                }
                asm volatile("barrier.sync 12, 128;" ::: "memory");
                float m_lane_2 = -CAKE_INF;
                float sink_lane_2 = -CAKE_INF;
                float m_scaled_2 = 0.0f;
                if (lane < 4) {
                    float _max_119 = max_noftz(smem_pmax[64 + lane], smem_pmax[72 + lane]);
                    float _max_120 = max_noftz(smem_pmax[80 + lane], smem_pmax[88 + lane]);
                    float _max_121 = max_noftz(_max_119, _max_120);
                    m_lane_2 = _max_121;
                    if (has_sinks != 0 && split_idx_2 == 0 && head_base_2 + 12 + lane < num_heads) {
                        sink_lane_2 = sinks[head_base_2 + 12 + lane] * 1.4426950408889634f;
                    }
                    float _max_122 = max_noftz(m_lane_2 * softmax_scale_log2_2, sink_lane_2);
                    m_scaled_2 = _max_122;
                    if (m_scaled_2 == -CAKE_INF) {
                        m_scaled_2 = 0.0f;
                    }
                }
                float col_max_2[4];
                float _shfl_75;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_75) : "f"(m_scaled_2), "r"(0));
                col_max_2[0] = _shfl_75;
                float _shfl_76;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_76) : "f"(m_scaled_2), "r"(1));
                col_max_2[1] = _shfl_76;
                float _shfl_77;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_77) : "f"(m_scaled_2), "r"(2));
                col_max_2[2] = _shfl_77;
                float _shfl_78;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_78) : "f"(m_scaled_2), "r"(3));
                col_max_2[3] = _shfl_78;
                float _exp2_14 = approx_exp2(score_values_2[0] * softmax_scale_log2_2 - col_max_2[0]);
                score_values_2[0] = _exp2_14;
                float _exp2_15 = approx_exp2(score_values_2[1] * softmax_scale_log2_2 - col_max_2[1]);
                score_values_2[1] = _exp2_15;
                float _exp2_16 = approx_exp2(score_values_2[2] * softmax_scale_log2_2 - col_max_2[2]);
                score_values_2[2] = _exp2_16;
                float _exp2_17 = approx_exp2(score_values_2[3] * softmax_scale_log2_2 - col_max_2[3]);
                score_values_2[3] = _exp2_17;
                {
                    uint16_t _fp8_pair_26;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_26) : "f"(0.0f), "f"(score_values_2[0]));
                    uint32_t _byte_26 = (uint32_t)(_fp8_pair_26 & 0xFF);
                    uint32_t _addr_26 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1536 + row_2 ^ (1536 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_26), "r"(_byte_26) : "memory");
                }
                float _fp8_rt_12;
                uint16_t _e4m3x2_27;
                uint32_t _f16x2_27;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_27) : "f"(0.0f), "f"(score_values_2[0]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_27) : "h"(_e4m3x2_27));
                uint16_t _fp8_h0_27 = (uint16_t)(_f16x2_27 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_12) : "h"(_fp8_h0_27));
                tr_vals_2[0] = _fp8_rt_12;
                {
                    uint16_t _fp8_pair_28;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_28) : "f"(0.0f), "f"(score_values_2[1]));
                    uint32_t _byte_28 = (uint32_t)(_fp8_pair_28 & 0xFF);
                    uint32_t _addr_28 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1664 + row_2 ^ (1664 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_28), "r"(_byte_28) : "memory");
                }
                float _fp8_rt_13;
                uint16_t _e4m3x2_29;
                uint32_t _f16x2_29;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_29) : "f"(0.0f), "f"(score_values_2[1]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_29) : "h"(_e4m3x2_29));
                uint16_t _fp8_h0_29 = (uint16_t)(_f16x2_29 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_13) : "h"(_fp8_h0_29));
                tr_vals_2[1] = _fp8_rt_13;
                {
                    uint16_t _fp8_pair_30;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_30) : "f"(0.0f), "f"(score_values_2[2]));
                    uint32_t _byte_30 = (uint32_t)(_fp8_pair_30 & 0xFF);
                    uint32_t _addr_30 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1792 + row_2 ^ (1792 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_30), "r"(_byte_30) : "memory");
                }
                float _fp8_rt_14;
                uint16_t _e4m3x2_31;
                uint32_t _f16x2_31;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_31) : "f"(0.0f), "f"(score_values_2[2]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_31) : "h"(_e4m3x2_31));
                uint16_t _fp8_h0_31 = (uint16_t)(_f16x2_31 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_14) : "h"(_fp8_h0_31));
                tr_vals_2[2] = _fp8_rt_14;
                {
                    uint16_t _fp8_pair_32;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                        : "=h"(_fp8_pair_32) : "f"(0.0f), "f"(score_values_2[3]));
                    uint32_t _byte_32 = (uint32_t)(_fp8_pair_32 & 0xFF);
                    uint32_t _addr_32 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(1920 + row_2 ^ (1920 + row_2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_32), "r"(_byte_32) : "memory");
                }
                float _fp8_rt_15;
                uint16_t _e4m3x2_33;
                uint32_t _f16x2_33;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_33) : "f"(0.0f), "f"(score_values_2[3]));
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_33) : "h"(_e4m3x2_33));
                uint16_t _fp8_h0_33 = (uint16_t)(_f16x2_33 & 0xFFFFu);
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_15) : "h"(_fp8_h0_33));
                tr_vals_2[3] = _fp8_rt_15;
                int hi_bit_9_2 = lane & 16;
                float send_10_2 = ((hi_bit_9_2 != 0) ? score_values_2[0] : score_values_2[2]);
                float keep_11_2 = ((hi_bit_9_2 != 0) ? score_values_2[2] : score_values_2[0]);
                float _shfl_79;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_79) : "f"(send_10_2), "r"(lane ^ 16));
                float recv_12_2 = _shfl_79;
                score_values_2[0] = keep_11_2 + recv_12_2;
                float send_13_2 = ((hi_bit_9_2 != 0) ? score_values_2[1] : score_values_2[3]);
                float keep_14_2 = ((hi_bit_9_2 != 0) ? score_values_2[3] : score_values_2[1]);
                float _shfl_80;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_80) : "f"(send_13_2), "r"(lane ^ 16));
                float recv_15_2 = _shfl_80;
                score_values_2[1] = keep_14_2 + recv_15_2;
                int hi_bit_16_2 = lane & 8;
                float send_17_2 = ((hi_bit_16_2 != 0) ? score_values_2[0] : score_values_2[1]);
                float keep_18_2 = ((hi_bit_16_2 != 0) ? score_values_2[1] : score_values_2[0]);
                float _shfl_81;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_81) : "f"(send_17_2), "r"(lane ^ 8));
                float recv_19_2 = _shfl_81;
                score_values_2[0] = keep_18_2 + recv_19_2;
                float _shfl_82;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_82) : "f"(score_values_2[0]), "r"(lane ^ 4));
                float other_20_2 = _shfl_82;
                score_values_2[0] = score_values_2[0] + other_20_2;
                float _shfl_83;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_83) : "f"(score_values_2[0]), "r"(lane ^ 2));
                float other_21_1 = _shfl_83;
                score_values_2[0] = score_values_2[0] + other_21_1;
                float _shfl_84;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_84) : "f"(score_values_2[0]), "r"(lane ^ 1));
                float other_22_1 = _shfl_84;
                score_values_2[0] = score_values_2[0] + other_22_1;
                int hi_bit_23_1 = lane & 16;
                float send_24_1 = ((hi_bit_23_1 != 0) ? tr_vals_2[0] : tr_vals_2[2]);
                float keep_25_1 = ((hi_bit_23_1 != 0) ? tr_vals_2[2] : tr_vals_2[0]);
                float _shfl_85;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_85) : "f"(send_24_1), "r"(lane ^ 16));
                float recv_26_1 = _shfl_85;
                tr_vals_2[0] = keep_25_1 + recv_26_1;
                float send_27_1 = ((hi_bit_23_1 != 0) ? tr_vals_2[1] : tr_vals_2[3]);
                float keep_28_1 = ((hi_bit_23_1 != 0) ? tr_vals_2[3] : tr_vals_2[1]);
                float _shfl_86;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_86) : "f"(send_27_1), "r"(lane ^ 16));
                float recv_29_1 = _shfl_86;
                tr_vals_2[1] = keep_28_1 + recv_29_1;
                int hi_bit_30_1 = lane & 8;
                float send_31_2 = ((hi_bit_30_1 != 0) ? tr_vals_2[0] : tr_vals_2[1]);
                float keep_32_2 = ((hi_bit_30_1 != 0) ? tr_vals_2[1] : tr_vals_2[0]);
                float _shfl_87;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_87) : "f"(send_31_2), "r"(lane ^ 8));
                float recv_33_2 = _shfl_87;
                tr_vals_2[0] = keep_32_2 + recv_33_2;
                float _shfl_88;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_88) : "f"(tr_vals_2[0]), "r"(lane ^ 4));
                float other_34_1 = _shfl_88;
                tr_vals_2[0] = tr_vals_2[0] + other_34_1;
                float _shfl_89;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_89) : "f"(tr_vals_2[0]), "r"(lane ^ 2));
                float other_35_1 = _shfl_89;
                tr_vals_2[0] = tr_vals_2[0] + other_35_1;
                float _shfl_90;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_90) : "f"(tr_vals_2[0]), "r"(lane ^ 1));
                float other_36_1 = _shfl_90;
                tr_vals_2[0] = tr_vals_2[0] + other_36_1;
                if ((lane & 7) == 0) {
                    smem_psum[(8 + local_warp_2) * 8 + (lane >> 3)] = score_values_2[0];
                    smem_rsum[(8 + local_warp_2) * 8 + (lane >> 3)] = tr_vals_2[0];
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                asm volatile("barrier.sync 12, 128;" ::: "memory");
                float norm_lane_2 = 0.0f;
                if (lane < 4) {
                    float _exp2_18 = approx_exp2(sink_lane_2 - m_scaled_2);
                    float sink_term_2 = _exp2_18;
                    float col_sum_2 = smem_psum[64 + lane] + smem_psum[72 + lane] + smem_psum[80 + lane] + smem_psum[88 + lane] + sink_term_2;
                    float denom_2 = smem_rsum[64 + lane] + smem_rsum[72 + lane] + smem_rsum[80 + lane] + smem_rsum[88 + lane] + sink_term_2;
                    if (denom_2 > 0.0f) {
                        float _rcp_2 = approx_rcp(denom_2);
                        norm_lane_2 = _rcp_2 * output_scale_2;
                    }
                    if (local_warp_2 == 0) {
                        if (o_chunk_2 == 0 && head_base_2 + 12 + lane < num_heads) {
                            int lse_offset_2 = (query_idx_2 * num_heads + head_base_2 + 12 + lane) * num_splits + split_idx_2;
                            float _log2_2;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(col_sum_2));
                            partial_lse[lse_offset_2] = ((col_sum_2 > 0.0f) ? (m_scaled_2 + _log2_2) * lse_partial_scale : -CAKE_INF);
                        }
                    }
                }
                float norm_c_2[4];
                float _shfl_91;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_91) : "f"(norm_lane_2), "r"(0));
                norm_c_2[0] = _shfl_91;
                float _shfl_92;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_92) : "f"(norm_lane_2), "r"(1));
                norm_c_2[1] = _shfl_92;
                float _shfl_93;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_93) : "f"(norm_lane_2), "r"(2));
                norm_c_2[2] = _shfl_93;
                float _shfl_94;
                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_94) : "f"(norm_lane_2), "r"(3));
                norm_c_2[3] = _shfl_94;
                float o_values_2[4];
                int dim_2 = 0;
                long long out_off_2 = 0;
                mbarrier_wait_hint(o_full_addr, _phase_o_full_0_2, 10000000);
                _phase_o_full_0_2 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x4(&o_values_2[0], taddr + 16 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_2 = o_chunk_2 * 4 * 128 + row_2;
                if (head_base_2 + 12 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[0] * norm_c_2[0];
                }
                if (head_base_2 + 12 + 1 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 1) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[1] * norm_c_2[1];
                }
                if (head_base_2 + 12 + 2 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 2) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[2] * norm_c_2[2];
                }
                if (head_base_2 + 12 + 3 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 3) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[3] * norm_c_2[3];
                }
                mbarrier_wait_hint(o_full_addr + 8, _phase_o_full_1_2, 10000000);
                _phase_o_full_1_2 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x4(&o_values_2[0], taddr + 32 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_2 = (o_chunk_2 * 4 + 1) * 128 + row_2;
                if (head_base_2 + 12 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[0] * norm_c_2[0];
                }
                if (head_base_2 + 12 + 1 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 1) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[1] * norm_c_2[1];
                }
                if (head_base_2 + 12 + 2 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 2) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[2] * norm_c_2[2];
                }
                if (head_base_2 + 12 + 3 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 3) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[3] * norm_c_2[3];
                }
                mbarrier_wait_hint(o_full_addr + 16, _phase_o_full_2_2, 10000000);
                _phase_o_full_2_2 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x4(&o_values_2[0], taddr + 48 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_2 = (o_chunk_2 * 4 + 2) * 128 + row_2;
                if (head_base_2 + 12 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[0] * norm_c_2[0];
                }
                if (head_base_2 + 12 + 1 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 1) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[1] * norm_c_2[1];
                }
                if (head_base_2 + 12 + 2 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 2) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[2] * norm_c_2[2];
                }
                if (head_base_2 + 12 + 3 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 3) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[3] * norm_c_2[3];
                }
                mbarrier_wait_hint(o_full_addr + 24, _phase_o_full_3_2, 10000000);
                _phase_o_full_3_2 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                tmem_ld_x4(&o_values_2[0], taddr + 64 + 12 + (unsigned int)(tmem_row_origin_2 << 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                dim_2 = (o_chunk_2 * 4 + 3) * 128 + row_2;
                if (head_base_2 + 12 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[0] * norm_c_2[0];
                }
                if (head_base_2 + 12 + 1 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 1) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[1] * norm_c_2[1];
                }
                if (head_base_2 + 12 + 2 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 2) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[2] * norm_c_2[2];
                }
                if (head_base_2 + 12 + 3 < num_heads) {
                    out_off_2 = ((long long)(query_idx_2 * num_heads + head_base_2 + 12 + 3) * (long long)num_splits + (long long)split_idx_2) * 512 + (long long)dim_2;
                    partial_O[out_off_2] = o_values_2[3] * norm_c_2[3];
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
                    "mov.b32 id, 134481040;\n\t"
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
                        0x8040480U, tmem_tmem_sfa0 + 0, tmem_tmem_sfb0 + 0, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                        0x8040480U, tmem_tmem_sfa0 + 4, tmem_tmem_sfb0 + 4, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                        0x8040480U, tmem_tmem_sfa0 + 8, tmem_tmem_sfb0 + 8, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                        0x8040480U, tmem_tmem_sfa0 + 12, tmem_tmem_sfb0 + 12, 1);
                }
                int _mma_a_lo_2 = (((smem_kf4_addr) >> 4) & 0x3FFF) + (1) * 1024;
                int _mma_b_lo_2 = (((smem_qf4b_addr) >> 4) & 0x3FFF) + (1) * 1024;
                {
                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 0, b_desc + 0,
                        0x8040480U, tmem_tmem_sfa1 + 0, tmem_tmem_sfb1 + 0, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 2, b_desc + 2,
                        0x8040480U, tmem_tmem_sfa1 + 4, tmem_tmem_sfb1 + 4, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 4, b_desc + 4,
                        0x8040480U, tmem_tmem_sfa1 + 8, tmem_tmem_sfb1 + 8, 1);
                    tcgen05_mma_mxf4nvf4_bs(tmem_tmem_s, a_desc + 6, b_desc + 6,
                        0x8040480U, tmem_tmem_sfa1 + 12, tmem_tmem_sfb1 + 12, 1);
                }
                tcgen05_commit(s_full_addr);
            }
            unsigned int _phase_p_full_0 = 0;
            mbarrier_wait_hint(p_full_addr, _phase_p_full_0, 10000000);
            _phase_p_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (elect_sync()) {
                int _mma_a_lo_3 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (0) * 1024;
                int _mma_b_lo_3 = (((smem_pb_addr) >> 4) & 0x3FFF) | 0x800000;
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
                    "mov.b32 id, 134512656;\n\t"
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
                int _mma_a_lo_4 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (1) * 1024;
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
                    "mov.b32 id, 134512656;\n\t"
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_3), "r"(tmem_tmem_o1), "r"(0));
                tcgen05_commit(o_full_addr + 8);
                int _mma_a_lo_5 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (2) * 1024;
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
                    "mov.b32 id, 134512656;\n\t"
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_3), "r"(tmem_tmem_o2), "r"(0));
                tcgen05_commit(o_full_addr + 16);
                int _mma_a_lo_6 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (3) * 1024;
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
                    "mov.b32 id, 134512656;\n\t"
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
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_3), "r"(tmem_tmem_o3), "r"(0));
                tcgen05_commit(o_full_addr + 24);
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
            int work_idx_3 = blockIdx.x;
            int head_tile_3 = work_idx_3 % num_head_tiles;
            int split_work_3 = work_idx_3 / num_head_tiles;
            int query_idx_3 = split_work_3 / num_splits;
            if (load_tid == 0) {
                mbarrier_arrive_expect_tx(q_nope_full0_addr, 6144);
                tma_4d_gmem2smem(smem_qstage_addr, (&tmap_q), 0, head_tile_3 * 128, 0, query_idx_3, q_nope_full0_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 2048, (&tmap_q), 0, head_tile_3 * 128, 1, query_idx_3, q_nope_full0_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 4096, (&tmap_q), 0, head_tile_3 * 128, 2, query_idx_3, q_nope_full0_addr);
                mbarrier_arrive_expect_tx(q_nope_full1_addr, 4096);
                tma_4d_gmem2smem(smem_qstage_addr + 6144, (&tmap_q), 0, head_tile_3 * 128, 3, query_idx_3, q_nope_full1_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 8192, (&tmap_q), 0, head_tile_3 * 128, 4, query_idx_3, q_nope_full1_addr);
                mbarrier_arrive_expect_tx(q_nope_full2_addr, 4096);
                tma_4d_gmem2smem(smem_qstage_addr + 10240, (&tmap_q), 0, head_tile_3 * 128, 5, query_idx_3, q_nope_full2_addr);
                tma_4d_gmem2smem(smem_qstage_addr + 12288, (&tmap_q), 0, head_tile_3 * 128, 6, query_idx_3, q_nope_full2_addr);
                mbarrier_arrive_expect_tx(q_rope_full_addr, 2048);
                tma_4d_gmem2smem(smem_qrope_addr, (&tmap_q), 0, head_tile_3 * 128, 7, query_idx_3, q_rope_full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
