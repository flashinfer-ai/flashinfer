/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) WanHybridTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) WanHybridTensorMapPack { WanHybridTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

#define WAN_HYBRID_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_SCORES_0_OFFSET 0
#define TMEM_SCORES_1_OFFSET 128
#define TMEM_TMEM_SFA_PV0_LO_OFFSET 80
#define TMEM_TMEM_SFA_PV0_HI_OFFSET 84
#define TMEM_TMEM_SFB_PV0_LO_OFFSET 88
#define TMEM_TMEM_SFB_PV0_HI_OFFSET 92
#define TMEM_TMEM_SFB_PV0_RES_LO_OFFSET 96
#define TMEM_TMEM_SFB_PV0_RES_HI_OFFSET 100
#define TMEM_TMEM_SFA_PV1_LO_OFFSET 208
#define TMEM_TMEM_SFA_PV1_HI_OFFSET 212
#define TMEM_TMEM_SFB_PV1_LO_OFFSET 216
#define TMEM_TMEM_SFB_PV1_HI_OFFSET 220
#define TMEM_TMEM_SFB_PV1_RES_LO_OFFSET 224
#define TMEM_TMEM_SFB_PV1_RES_HI_OFFSET 228
#define TMEM_OUTPUT_0_OFFSET 256
#define TMEM_OUTPUT_1_OFFSET 384
#define NUM_Q_STAGES 2
#define NUM_KV_STAGES 3
#define NUM_ACC_STAGES 1
#define SMEM_SSCALE_OFF 1024
#define SMEM_SSCALE_STAGE_BYTES 1024
#define SMEM_SSCALE_STRIDE 1024
#define SMEM_SMEM_Q0_OFF 2048
#define SMEM_SMEM_Q0_STAGE_BYTES 32768
#define SMEM_SMEM_Q0_STRIDE 32768
#define SMEM_SMEM_Q1_OFF 34816
#define SMEM_SMEM_Q1_STAGE_BYTES 32768
#define SMEM_SMEM_Q1_STRIDE 32768
#define SMEM_SMEM_KV_OFF 67584
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_VT_OFF 67584
#define SMEM_SMEM_VT_STAGE_BYTES 8192
#define SMEM_SMEM_VT_STRIDE 32768
#define SMEM_SMEM_VT_RESIDUAL_OFF 75776
#define SMEM_SMEM_VT_RESIDUAL_STAGE_BYTES 8192
#define SMEM_SMEM_VT_RESIDUAL_STRIDE 32768
#define SMEM_SMEM_SFVT_LO_OFF 83968
#define SMEM_SMEM_SFVT_LO_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_LO_STRIDE 32768
#define SMEM_SMEM_SFVT_HI_OFF 84480
#define SMEM_SMEM_SFVT_HI_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_HI_STRIDE 32768
#define SMEM_SMEM_SFVT_RESIDUAL_LO_OFF 84992
#define SMEM_SMEM_SFVT_RESIDUAL_LO_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_RESIDUAL_LO_STRIDE 32768
#define SMEM_SMEM_SFVT_RESIDUAL_HI_OFF 85504
#define SMEM_SMEM_SFVT_RESIDUAL_HI_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_RESIDUAL_HI_STRIDE 32768
#define SMEM_SMEM_O_OFF 165888
#define SMEM_SMEM_O_STAGE_BYTES 32768
#define SMEM_SMEM_O_STRIDE 32768
#define SMEM_TOTAL 231424
#define IS_CAUSAL 0
#define HAS_TAIL 1
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128
#define MMA_K 16
#define NUM_M_BLOCKS (((seqlen_q + (2 * BLOCK_M)) - 1) / (2 * BLOCK_M))

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
        :: "r"(mbar_addr), "r"(count));
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

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_ts(
    int taddr, int a_taddr, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], [%1], %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "r"(a_taddr), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
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


__device__ __forceinline__ void tmem_ld_x32(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(tmem_addr));
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


__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(*x0_ptr, -127.0f), x1 = max_noftz(*x1_ptr, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    *x0_ptr = r0; *x1_ptr = r1;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
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
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
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

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


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


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo128(int addr) {
    const int SBO = 128;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_5d(
    const void *tmap, int x, int y, int z, int w, int v, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4, %5}], [%6];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_wan_hybrid_attention(WanHybridTensorMap const* Q, WanHybridTensorMap const* K, WanHybridTensorMap const* Vt, WanHybridTensorMap const* SFVtLo, WanHybridTensorMap const* SFVtHi, WanHybridTensorMap const* O, int seqlen_q, int seqlen_kv, float softmax_scale_log2, int heads, int total_bh, int physical_num_blocks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Vt)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFVtLo)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFVtHi)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(O)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* sScale = reinterpret_cast<float*>(smem_raw + 1024);
    const int sScale_addr = smem + 1024;
    __nv_bfloat16* smem_q0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_q0_addr = smem + 2048;
    __nv_bfloat16* smem_q1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 34816);
    const int smem_q1_addr = smem + 34816;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 67584);
    const int smem_kv_addr = smem + 67584;
    uint8_t* smem_vt = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_vt_addr = smem + 67584;
    uint8_t* smem_vt_residual = reinterpret_cast<uint8_t*>(smem_raw + 75776);
    const int smem_vt_residual_addr = smem + 75776;
    uint8_t* smem_sfvt_lo = reinterpret_cast<uint8_t*>(smem_raw + 83968);
    const int smem_sfvt_lo_addr = smem + 83968;
    uint8_t* smem_sfvt_hi = reinterpret_cast<uint8_t*>(smem_raw + 84480);
    const int smem_sfvt_hi_addr = smem + 84480;
    uint8_t* smem_sfvt_residual_lo = reinterpret_cast<uint8_t*>(smem_raw + 84992);
    const int smem_sfvt_residual_lo_addr = smem + 84992;
    uint8_t* smem_sfvt_residual_hi = reinterpret_cast<uint8_t*>(smem_raw + 85504);
    const int smem_sfvt_residual_hi_addr = smem + 85504;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 165888);
    const int smem_o_addr = smem + 165888;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Q)) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Vt)) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(SFVtLo)) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(SFVtHi)) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(O)) : "memory"); }

    // Mbarrier init (11 groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    {
        const int warp = make_warp_uniform(tid / 32);
        if (warp == 0) {
            uint32_t leader = elect_sync();
            if (leader) {
                // q_full: 2 barriers, init_count=1
                mbarrier_init(smem + 0, 1);
                mbarrier_init(smem + 8, 1);
                // q_empty: 2 barriers, init_count=1
                mbarrier_init(smem + 16, 1);
                mbarrier_init(smem + 24, 1);
                // --- pipeline 'kv' ---
                // kv_full: 3 barriers, init_count=1
                mbarrier_init(smem + 32, 1);
                mbarrier_init(smem + 40, 1);
                mbarrier_init(smem + 48, 1);
                // kv_empty: 3 barriers, init_count=1
                mbarrier_init(smem + 56, 1);
                mbarrier_init(smem + 64, 1);
                mbarrier_init(smem + 72, 1);
                // p_full: 2 barriers, init_count=256
                mbarrier_init(smem + 80, 256);
                mbarrier_init(smem + 88, 256);
                // s_full: 2 barriers, init_count=1
                mbarrier_init(smem + 96, 1);
                mbarrier_init(smem + 104, 1);
                // o_full: 2 barriers, init_count=1
                mbarrier_init(smem + 112, 1);
                mbarrier_init(smem + 120, 1);
                // corr_sig: 2 barriers, init_count=128
                mbarrier_init(smem + 128, 128);
                mbarrier_init(smem + 136, 128);
                // p_full_2: 2 barriers, init_count=128
                mbarrier_init(smem + 144, 128);
                mbarrier_init(smem + 152, 128);
                // epi_full: 2 barriers, init_count=128
                mbarrier_init(smem + 160, 128);
                mbarrier_init(smem + 168, 128);
                // epi_empty: 2 barriers, init_count=32
                mbarrier_init(smem + 176, 32);
                mbarrier_init(smem + 184, 32);
                asm volatile("fence.mbarrier_init.release.cluster;");
            }
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    {
        const int warp = make_warp_uniform(tid / 32);
        if (warp == 12) {
            int _tmem_hold = smem + 192;
            asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        }
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 56)
    #define p_full_addr (mbar_base + 80)
    #define s_full_addr (mbar_base + 96)
    #define o_full_addr (mbar_base + 112)
    #define corr_sig_addr (mbar_base + 128)
    #define p_full_2_addr (mbar_base + 144)
    #define epi_full_addr (mbar_base + 160)
    #define epi_empty_addr (mbar_base + 176)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores_0 = taddr;
    const int tmem_scores_1 = taddr + 128;
    const int tmem_tmem_sfa_pv0_lo = taddr + 80;
    const int tmem_tmem_sfa_pv0_hi = taddr + 84;
    const int tmem_tmem_sfb_pv0_lo = taddr + 88;
    const int tmem_tmem_sfb_pv0_hi = taddr + 92;
    const int tmem_tmem_sfb_pv0_res_lo = taddr + 96;
    const int tmem_tmem_sfb_pv0_res_hi = taddr + 100;
    const int tmem_tmem_sfa_pv1_lo = taddr + 208;
    const int tmem_tmem_sfa_pv1_hi = taddr + 212;
    const int tmem_tmem_sfb_pv1_lo = taddr + 216;
    const int tmem_tmem_sfb_pv1_hi = taddr + 220;
    const int tmem_tmem_sfb_pv1_res_lo = taddr + 224;
    const int tmem_tmem_sfb_pv1_res_hi = taddr + 228;
    const int tmem_output_0 = taddr + 256;
    const int tmem_output_1 = taddr + 384;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    {
        const int warp = make_warp_uniform(tid / 32);
        if (warp >= 12 && warp <= 15) {
            asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        }
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_main
            unsigned int stage = make_warp_uniform(warp / 4);
            int tmem_s_off = make_warp_uniform(stage * 128);
            int tmem_p_off = make_warp_uniform(stage * 128 + 64);
            int scale_off = make_warp_uniform(stage * (unsigned int)BLOCK_M);
            unsigned int total_tiles = NUM_M_BLOCKS * total_bh;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_o_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < total_tiles; tile_idx += num_bids) {
                unsigned int m_block;
                unsigned int bh;
                {
                    m_block = tile_idx % (unsigned int)NUM_M_BLOCKS;
                    bh = tile_idx / (unsigned int)NUM_M_BLOCKS;
                }
                unsigned int num_n_blocks = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                int causal_row;
                unsigned int num_masked_iters;
                {
                    num_masked_iters = 0;
                }
                float row_max_val = -WAN_HYBRID_INF;
                float row_sum_val = 0.0f;
                int n_block = num_n_blocks - 1;
                mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                _phase_s_full ^= 1;
                int s_base = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                float _tmem_load_0[128];
                tmem_ld_x32(&_tmem_load_0[0], s_base);
                tmem_ld_x32(&_tmem_load_0[32], s_base + 32);
                tmem_ld_x32(&_tmem_load_0[64], s_base + 64);
                tmem_ld_x32(&_tmem_load_0[96], s_base + 96);
                {
                    int tail_valid = seqlen_kv - n_block * BLOCK_N;
                    if (tail_valid < BLOCK_N) {
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_0 = tail_valid;
                            if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                            else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                            }
                        }
                        #pragma unroll
                        for (int _i_1 = 0; _i_1 < 32; _i_1++) {
                            if (!(_slice_lo_mask_0 & (1u << _i_1))) _tmem_load_0[0 + _i_1] = -WAN_HYBRID_INF;
                        }
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_2 = tail_valid - 32;
                            if (_lim_2 <= 0) { _slice_lo_mask_1 = 0u; }
                            else if (_lim_2 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_2));
                            }
                        }
                        #pragma unroll
                        for (int _i_3 = 0; _i_3 < 32; _i_3++) {
                            if (!(_slice_lo_mask_1 & (1u << _i_3))) _tmem_load_0[32 + _i_3] = -WAN_HYBRID_INF;
                        }
                        uint32_t _slice_lo_mask_2;
                        {
                            int _lim_4 = tail_valid - 64;
                            if (_lim_4 <= 0) { _slice_lo_mask_2 = 0u; }
                            else if (_lim_4 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_4));
                            }
                        }
                        #pragma unroll
                        for (int _i_5 = 0; _i_5 < 32; _i_5++) {
                            if (!(_slice_lo_mask_2 & (1u << _i_5))) _tmem_load_0[64 + _i_5] = -WAN_HYBRID_INF;
                        }
                        uint32_t _slice_lo_mask_3;
                        {
                            int _lim_6 = tail_valid - 96;
                            if (_lim_6 <= 0) { _slice_lo_mask_3 = 0u; }
                            else if (_lim_6 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_6));
                            }
                        }
                        #pragma unroll
                        for (int _i_7 = 0; _i_7 < 32; _i_7++) {
                            if (!(_slice_lo_mask_3 & (1u << _i_7))) _tmem_load_0[96 + _i_7] = -WAN_HYBRID_INF;
                        }
                    }
                }
                float new_max = -WAN_HYBRID_INF;
                float group_max0 = -WAN_HYBRID_INF;
                float group_max1 = -WAN_HYBRID_INF;
                float group_max2 = -WAN_HYBRID_INF;
                float group_max3 = -WAN_HYBRID_INF;
                float group_max4 = -WAN_HYBRID_INF;
                float group_max5 = -WAN_HYBRID_INF;
                float group_max6 = -WAN_HYBRID_INF;
                float group_max7 = -WAN_HYBRID_INF;
                float _max3_0;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_0) : "f"(_tmem_load_0[0]), "f"(_tmem_load_0[1]), "f"(_tmem_load_0[2]));
                float max012 = _max3_0;
                float _max3_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_1) : "f"(_tmem_load_0[3]), "f"(_tmem_load_0[4]), "f"(_tmem_load_0[5]));
                float max345 = _max3_1;
                float _max3_2;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_2) : "f"(_tmem_load_0[6]), "f"(_tmem_load_0[7]), "f"(_tmem_load_0[8]));
                float max678 = _max3_2;
                float _max3_3;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_3) : "f"(_tmem_load_0[9]), "f"(_tmem_load_0[10]), "f"(_tmem_load_0[11]));
                float max9ab = _max3_3;
                float _max3_4;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_4) : "f"(_tmem_load_0[12]), "f"(_tmem_load_0[13]), "f"(_tmem_load_0[14]));
                float maxcde = _max3_4;
                float _max3_5;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_5) : "f"(max012), "f"(max345), "f"(max678));
                float max0_8 = _max3_5;
                float _max3_6;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_6) : "f"(max9ab), "f"(maxcde), "f"(_tmem_load_0[15]));
                float max9_f = _max3_6;
                float _max_0 = max_noftz(max0_8, max9_f);
                float group_max0_0 = _max_0;
                float _max3_7;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_7) : "f"(_tmem_load_0[16]), "f"(_tmem_load_0[17]), "f"(_tmem_load_0[18]));
                float max012_1 = _max3_7;
                float _max3_8;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_8) : "f"(_tmem_load_0[19]), "f"(_tmem_load_0[20]), "f"(_tmem_load_0[21]));
                float max345_2 = _max3_8;
                float _max3_9;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_9) : "f"(_tmem_load_0[22]), "f"(_tmem_load_0[23]), "f"(_tmem_load_0[24]));
                float max678_3 = _max3_9;
                float _max3_10;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_10) : "f"(_tmem_load_0[25]), "f"(_tmem_load_0[26]), "f"(_tmem_load_0[27]));
                float max9ab_4 = _max3_10;
                float _max3_11;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_11) : "f"(_tmem_load_0[28]), "f"(_tmem_load_0[29]), "f"(_tmem_load_0[30]));
                float maxcde_5 = _max3_11;
                float _max3_12;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_12) : "f"(max012_1), "f"(max345_2), "f"(max678_3));
                float max0_8_6 = _max3_12;
                float _max3_13;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_13) : "f"(max9ab_4), "f"(maxcde_5), "f"(_tmem_load_0[31]));
                float max9_f_7 = _max3_13;
                float _max_1 = max_noftz(max0_8_6, max9_f_7);
                float group_max1_8 = _max_1;
                float _max3_14;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_14) : "f"(_tmem_load_0[32]), "f"(_tmem_load_0[33]), "f"(_tmem_load_0[34]));
                float max012_9 = _max3_14;
                float _max3_15;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_15) : "f"(_tmem_load_0[35]), "f"(_tmem_load_0[36]), "f"(_tmem_load_0[37]));
                float max345_10 = _max3_15;
                float _max3_16;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_16) : "f"(_tmem_load_0[38]), "f"(_tmem_load_0[39]), "f"(_tmem_load_0[40]));
                float max678_11 = _max3_16;
                float _max3_17;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_17) : "f"(_tmem_load_0[41]), "f"(_tmem_load_0[42]), "f"(_tmem_load_0[43]));
                float max9ab_12 = _max3_17;
                float _max3_18;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_18) : "f"(_tmem_load_0[44]), "f"(_tmem_load_0[45]), "f"(_tmem_load_0[46]));
                float maxcde_13 = _max3_18;
                float _max3_19;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_19) : "f"(max012_9), "f"(max345_10), "f"(max678_11));
                float max0_8_14 = _max3_19;
                float _max3_20;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_20) : "f"(max9ab_12), "f"(maxcde_13), "f"(_tmem_load_0[47]));
                float max9_f_15 = _max3_20;
                float _max_2 = max_noftz(max0_8_14, max9_f_15);
                float group_max2_16 = _max_2;
                float _max3_21;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_21) : "f"(_tmem_load_0[48]), "f"(_tmem_load_0[49]), "f"(_tmem_load_0[50]));
                float max012_17 = _max3_21;
                float _max3_22;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_22) : "f"(_tmem_load_0[51]), "f"(_tmem_load_0[52]), "f"(_tmem_load_0[53]));
                float max345_18 = _max3_22;
                float _max3_23;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_23) : "f"(_tmem_load_0[54]), "f"(_tmem_load_0[55]), "f"(_tmem_load_0[56]));
                float max678_19 = _max3_23;
                float _max3_24;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_24) : "f"(_tmem_load_0[57]), "f"(_tmem_load_0[58]), "f"(_tmem_load_0[59]));
                float max9ab_20 = _max3_24;
                float _max3_25;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(_tmem_load_0[60]), "f"(_tmem_load_0[61]), "f"(_tmem_load_0[62]));
                float maxcde_21 = _max3_25;
                float _max3_26;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_26) : "f"(max012_17), "f"(max345_18), "f"(max678_19));
                float max0_8_22 = _max3_26;
                float _max3_27;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_27) : "f"(max9ab_20), "f"(maxcde_21), "f"(_tmem_load_0[63]));
                float max9_f_23 = _max3_27;
                float _max_3 = max_noftz(max0_8_22, max9_f_23);
                float group_max3_24 = _max_3;
                float _max3_28;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_28) : "f"(_tmem_load_0[64]), "f"(_tmem_load_0[65]), "f"(_tmem_load_0[66]));
                float max012_25 = _max3_28;
                float _max3_29;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_29) : "f"(_tmem_load_0[67]), "f"(_tmem_load_0[68]), "f"(_tmem_load_0[69]));
                float max345_26 = _max3_29;
                float _max3_30;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_30) : "f"(_tmem_load_0[70]), "f"(_tmem_load_0[71]), "f"(_tmem_load_0[72]));
                float max678_27 = _max3_30;
                float _max3_31;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_31) : "f"(_tmem_load_0[73]), "f"(_tmem_load_0[74]), "f"(_tmem_load_0[75]));
                float max9ab_28 = _max3_31;
                float _max3_32;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_32) : "f"(_tmem_load_0[76]), "f"(_tmem_load_0[77]), "f"(_tmem_load_0[78]));
                float maxcde_29 = _max3_32;
                float _max3_33;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_33) : "f"(max012_25), "f"(max345_26), "f"(max678_27));
                float max0_8_30 = _max3_33;
                float _max3_34;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_34) : "f"(max9ab_28), "f"(maxcde_29), "f"(_tmem_load_0[79]));
                float max9_f_31 = _max3_34;
                float _max_4 = max_noftz(max0_8_30, max9_f_31);
                float group_max4_32 = _max_4;
                float _max3_35;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_35) : "f"(_tmem_load_0[80]), "f"(_tmem_load_0[81]), "f"(_tmem_load_0[82]));
                float max012_33 = _max3_35;
                float _max3_36;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_36) : "f"(_tmem_load_0[83]), "f"(_tmem_load_0[84]), "f"(_tmem_load_0[85]));
                float max345_34 = _max3_36;
                float _max3_37;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_37) : "f"(_tmem_load_0[86]), "f"(_tmem_load_0[87]), "f"(_tmem_load_0[88]));
                float max678_35 = _max3_37;
                float _max3_38;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_38) : "f"(_tmem_load_0[89]), "f"(_tmem_load_0[90]), "f"(_tmem_load_0[91]));
                float max9ab_36 = _max3_38;
                float _max3_39;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_39) : "f"(_tmem_load_0[92]), "f"(_tmem_load_0[93]), "f"(_tmem_load_0[94]));
                float maxcde_37 = _max3_39;
                float _max3_40;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_40) : "f"(max012_33), "f"(max345_34), "f"(max678_35));
                float max0_8_38 = _max3_40;
                float _max3_41;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_41) : "f"(max9ab_36), "f"(maxcde_37), "f"(_tmem_load_0[95]));
                float max9_f_39 = _max3_41;
                float _max_5 = max_noftz(max0_8_38, max9_f_39);
                float group_max5_40 = _max_5;
                float _max3_42;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_42) : "f"(_tmem_load_0[96]), "f"(_tmem_load_0[97]), "f"(_tmem_load_0[98]));
                float max012_41 = _max3_42;
                float _max3_43;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_43) : "f"(_tmem_load_0[99]), "f"(_tmem_load_0[100]), "f"(_tmem_load_0[101]));
                float max345_42 = _max3_43;
                float _max3_44;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_44) : "f"(_tmem_load_0[102]), "f"(_tmem_load_0[103]), "f"(_tmem_load_0[104]));
                float max678_43 = _max3_44;
                float _max3_45;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_45) : "f"(_tmem_load_0[105]), "f"(_tmem_load_0[106]), "f"(_tmem_load_0[107]));
                float max9ab_44 = _max3_45;
                float _max3_46;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_46) : "f"(_tmem_load_0[108]), "f"(_tmem_load_0[109]), "f"(_tmem_load_0[110]));
                float maxcde_45 = _max3_46;
                float _max3_47;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_47) : "f"(max012_41), "f"(max345_42), "f"(max678_43));
                float max0_8_46 = _max3_47;
                float _max3_48;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_48) : "f"(max9ab_44), "f"(maxcde_45), "f"(_tmem_load_0[111]));
                float max9_f_47 = _max3_48;
                float _max_6 = max_noftz(max0_8_46, max9_f_47);
                float group_max6_48 = _max_6;
                float _max3_49;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_49) : "f"(_tmem_load_0[112]), "f"(_tmem_load_0[113]), "f"(_tmem_load_0[114]));
                float max012_49 = _max3_49;
                float _max3_50;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_50) : "f"(_tmem_load_0[115]), "f"(_tmem_load_0[116]), "f"(_tmem_load_0[117]));
                float max345_50 = _max3_50;
                float _max3_51;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_51) : "f"(_tmem_load_0[118]), "f"(_tmem_load_0[119]), "f"(_tmem_load_0[120]));
                float max678_51 = _max3_51;
                float _max3_52;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_52) : "f"(_tmem_load_0[121]), "f"(_tmem_load_0[122]), "f"(_tmem_load_0[123]));
                float max9ab_52 = _max3_52;
                float _max3_53;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_53) : "f"(_tmem_load_0[124]), "f"(_tmem_load_0[125]), "f"(_tmem_load_0[126]));
                float maxcde_53 = _max3_53;
                float _max3_54;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_54) : "f"(max012_49), "f"(max345_50), "f"(max678_51));
                float max0_8_54 = _max3_54;
                float _max3_55;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_55) : "f"(max9ab_52), "f"(maxcde_53), "f"(_tmem_load_0[127]));
                float max9_f_55 = _max3_55;
                float _max_7 = max_noftz(max0_8_54, max9_f_55);
                float group_max7_56 = _max_7;
                float _max3_56;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_56) : "f"(group_max0_0), "f"(group_max1_8), "f"(group_max2_16));
                float max012_57 = _max3_56;
                float _max3_57;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_57) : "f"(group_max3_24), "f"(group_max4_32), "f"(group_max5_40));
                float max345_58 = _max3_57;
                float _max_8 = max_noftz(group_max6_48, group_max7_56);
                float max67 = _max_8;
                float _max3_58;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_58) : "f"(max012_57), "f"(max345_58), "f"(max67));
                new_max = _max3_58;
                group_max0 = group_max0_0;
                group_max1 = group_max1_8;
                group_max2 = group_max2_16;
                group_max3 = group_max3_24;
                group_max4 = group_max4_32;
                group_max5 = group_max5_40;
                group_max6 = group_max6_48;
                group_max7 = group_max7_56;
                row_max_val = new_max;
                float new_max_scaled = ((new_max == -WAN_HYBRID_INF) ? 0.0f : new_max) * softmax_scale_log2;
                mbarrier_arrive(corr_sig_addr + (stage) * 8);
                float2 _f2_0 = make_float2(0.0f, 0.0f);
                float2 block_sum2 = _f2_0;
                float sf_values[4];
                int p_stage_off = stage * 128;
                float block_max = group_max0;
                float block_max_scaled = ((block_max > -WAN_HYBRID_INF) ? block_max * softmax_scale_log2 : 0.0f);
                float _exp2_0 = approx_exp2(block_max_scaled - new_max_scaled - 2.584962500721156f);
                float p_scale = ((block_max > -WAN_HYBRID_INF) ? _exp2_0 : 0.0f);
                sf_values[0] = p_scale;
                const float2 _fma_b2_8 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_9 = {2.584962500721156f - block_max_scaled, 2.584962500721156f - block_max_scaled};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_0 + 0))[_lf], _fma_b2_8, _fma_c2_9);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                }
                float2 _f2_1 = make_float2(_tmem_load_0[0], _tmem_load_0[1]);
                float2 partial = _f2_1;
                #pragma unroll
                for (int pair = 2; pair < 16; pair += 2) {
                    float2 _f2_2 = make_float2((_tmem_load_0 + 0)[pair], (_tmem_load_0 + 0)[pair + 1]);
                    partial = add_f32x2(partial, _f2_2);
                }
                float2 frag_sum2 = partial;
                float2 _f2_3 = make_float2(p_scale, p_scale);
                float2 raw_scale2 = _f2_3;
                float2 raw_sum2 = mul_f32x2(frag_sum2, raw_scale2);
                uint32_t _fp4_0[2];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(_tmem_load_0[0]), "f"(_tmem_load_0[1]), "f"(_tmem_load_0[2]), "f"(_tmem_load_0[3]), "f"(_tmem_load_0[4]), "f"(_tmem_load_0[5]), "f"(_tmem_load_0[6]), "f"(_tmem_load_0[7]));
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[1]) : "f"(_tmem_load_0[8]), "f"(_tmem_load_0[9]), "f"(_tmem_load_0[10]), "f"(_tmem_load_0[11]), "f"(_tmem_load_0[12]), "f"(_tmem_load_0[13]), "f"(_tmem_load_0[14]), "f"(_tmem_load_0[15]));
                block_sum2 = add_f32x2(block_sum2, raw_sum2);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x2.b32"
                    " [%0], {%1, %2};"
                    :: "r"(taddr + (unsigned int)p_stage_off + 64 + (unsigned int)(warp % 4 * 32 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_0[1]))
                    : "memory");
                float block_max_59 = group_max1;
                float block_max_scaled_60 = ((block_max_59 > -WAN_HYBRID_INF) ? block_max_59 * softmax_scale_log2 : 0.0f);
                float _exp2_1 = approx_exp2(block_max_scaled_60 - new_max_scaled - 2.584962500721156f);
                float p_scale_61 = ((block_max_59 > -WAN_HYBRID_INF) ? _exp2_1 : 0.0f);
                sf_values[1] = p_scale_61;
                const float2 _fma_b2_10 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_11 = {2.584962500721156f - block_max_scaled_60, 2.584962500721156f - block_max_scaled_60};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_0 + 16))[_lf], _fma_b2_10, _fma_c2_11);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    _tmem_load_0[_le + 16] = approx_exp2(_tmem_load_0[_le + 16]);
                }
                float2 _f2_4 = make_float2(_tmem_load_0[16], _tmem_load_0[17]);
                float2 partial_62 = _f2_4;
                #pragma unroll
                for (int pair_1 = 2; pair_1 < 16; pair_1 += 2) {
                    float2 _f2_5 = make_float2((_tmem_load_0 + 16)[pair_1], (_tmem_load_0 + 16)[pair_1 + 1]);
                    partial_62 = add_f32x2(partial_62, _f2_5);
                }
                float2 frag_sum2_63 = partial_62;
                float2 _f2_6 = make_float2(p_scale_61, p_scale_61);
                float2 raw_scale2_64 = _f2_6;
                float2 raw_sum2_65 = mul_f32x2(frag_sum2_63, raw_scale2_64);
                uint32_t _fp4_1[2];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[0]) : "f"(_tmem_load_0[16]), "f"(_tmem_load_0[17]), "f"(_tmem_load_0[18]), "f"(_tmem_load_0[19]), "f"(_tmem_load_0[20]), "f"(_tmem_load_0[21]), "f"(_tmem_load_0[22]), "f"(_tmem_load_0[23]));
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[1]) : "f"(_tmem_load_0[24]), "f"(_tmem_load_0[25]), "f"(_tmem_load_0[26]), "f"(_tmem_load_0[27]), "f"(_tmem_load_0[28]), "f"(_tmem_load_0[29]), "f"(_tmem_load_0[30]), "f"(_tmem_load_0[31]));
                block_sum2 = add_f32x2(block_sum2, raw_sum2_65);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x2.b32"
                    " [%0], {%1, %2};"
                    :: "r"(taddr + (unsigned int)p_stage_off + 64 + (unsigned int)(warp % 4 * 32 << 16) + 2), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_1[1]))
                    : "memory");
                float block_max_66 = group_max2;
                float block_max_scaled_67 = ((block_max_66 > -WAN_HYBRID_INF) ? block_max_66 * softmax_scale_log2 : 0.0f);
                float _exp2_2 = approx_exp2(block_max_scaled_67 - new_max_scaled - 2.584962500721156f);
                float p_scale_68 = ((block_max_66 > -WAN_HYBRID_INF) ? _exp2_2 : 0.0f);
                sf_values[2] = p_scale_68;
                const float2 _fma_b2_12 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_13 = {2.584962500721156f - block_max_scaled_67, 2.584962500721156f - block_max_scaled_67};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_0 + 32))[_lf], _fma_b2_12, _fma_c2_13);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    _tmem_load_0[_le + 32] = approx_exp2(_tmem_load_0[_le + 32]);
                }
                float2 _f2_7 = make_float2(_tmem_load_0[32], _tmem_load_0[33]);
                float2 partial_69 = _f2_7;
                #pragma unroll
                for (int pair_2 = 2; pair_2 < 16; pair_2 += 2) {
                    float2 _f2_8 = make_float2((_tmem_load_0 + 32)[pair_2], (_tmem_load_0 + 32)[pair_2 + 1]);
                    partial_69 = add_f32x2(partial_69, _f2_8);
                }
                float2 frag_sum2_70 = partial_69;
                float2 _f2_9 = make_float2(p_scale_68, p_scale_68);
                float2 raw_scale2_71 = _f2_9;
                float2 raw_sum2_72 = mul_f32x2(frag_sum2_70, raw_scale2_71);
                uint32_t _fp4_2[2];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[0]) : "f"(_tmem_load_0[32]), "f"(_tmem_load_0[33]), "f"(_tmem_load_0[34]), "f"(_tmem_load_0[35]), "f"(_tmem_load_0[36]), "f"(_tmem_load_0[37]), "f"(_tmem_load_0[38]), "f"(_tmem_load_0[39]));
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[1]) : "f"(_tmem_load_0[40]), "f"(_tmem_load_0[41]), "f"(_tmem_load_0[42]), "f"(_tmem_load_0[43]), "f"(_tmem_load_0[44]), "f"(_tmem_load_0[45]), "f"(_tmem_load_0[46]), "f"(_tmem_load_0[47]));
                block_sum2 = add_f32x2(block_sum2, raw_sum2_72);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x2.b32"
                    " [%0], {%1, %2};"
                    :: "r"(taddr + (unsigned int)p_stage_off + 64 + (unsigned int)(warp % 4 * 32 << 16) + 4), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_2[1]))
                    : "memory");
                float block_max_73 = group_max3;
                float block_max_scaled_74 = ((block_max_73 > -WAN_HYBRID_INF) ? block_max_73 * softmax_scale_log2 : 0.0f);
                float _exp2_3 = approx_exp2(block_max_scaled_74 - new_max_scaled - 2.584962500721156f);
                float p_scale_75 = ((block_max_73 > -WAN_HYBRID_INF) ? _exp2_3 : 0.0f);
                sf_values[3] = p_scale_75;
                const float2 _fma_b2_14 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_15 = {2.584962500721156f - block_max_scaled_74, 2.584962500721156f - block_max_scaled_74};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_0 + 48))[_lf], _fma_b2_14, _fma_c2_15);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    _tmem_load_0[_le + 48] = approx_exp2(_tmem_load_0[_le + 48]);
                }
                float2 _f2_10 = make_float2(_tmem_load_0[48], _tmem_load_0[49]);
                float2 partial_76 = _f2_10;
                #pragma unroll
                for (int pair_3 = 2; pair_3 < 16; pair_3 += 2) {
                    float2 _f2_11 = make_float2((_tmem_load_0 + 48)[pair_3], (_tmem_load_0 + 48)[pair_3 + 1]);
                    partial_76 = add_f32x2(partial_76, _f2_11);
                }
                float2 frag_sum2_77 = partial_76;
                float2 _f2_12 = make_float2(p_scale_75, p_scale_75);
                float2 raw_scale2_78 = _f2_12;
                float2 raw_sum2_79 = mul_f32x2(frag_sum2_77, raw_scale2_78);
                uint32_t _fp4_3[2];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[0]) : "f"(_tmem_load_0[48]), "f"(_tmem_load_0[49]), "f"(_tmem_load_0[50]), "f"(_tmem_load_0[51]), "f"(_tmem_load_0[52]), "f"(_tmem_load_0[53]), "f"(_tmem_load_0[54]), "f"(_tmem_load_0[55]));
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[1]) : "f"(_tmem_load_0[56]), "f"(_tmem_load_0[57]), "f"(_tmem_load_0[58]), "f"(_tmem_load_0[59]), "f"(_tmem_load_0[60]), "f"(_tmem_load_0[61]), "f"(_tmem_load_0[62]), "f"(_tmem_load_0[63]));
                block_sum2 = add_f32x2(block_sum2, raw_sum2_79);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x2.b32"
                    " [%0], {%1, %2};"
                    :: "r"(taddr + (unsigned int)p_stage_off + 64 + (unsigned int)(warp % 4 * 32 << 16) + 6), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_3[1]))
                    : "memory");
                uint32_t _fp8_0[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(sf_values[0]), "f"(sf_values[1]),
                                           "f"(sf_values[2]), "f"(sf_values[3]));
                    _fp8_0[0] = _packed;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)p_stage_off + 80 + (unsigned int)(warp % 4)), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[0]))
                    : "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_addr + (stage) * 8);
                float _exp2_4 = approx_exp2(((group_max4 > -WAN_HYBRID_INF) ? group_max4 * softmax_scale_log2 : 0.0f) - new_max_scaled - 2.584962500721156f);
                sf_values[0] = ((group_max4 > -WAN_HYBRID_INF) ? _exp2_4 : 0.0f);
                const float2 _fma_b2_16 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_17 = {2.584962500721156f - ((group_max4 > -WAN_HYBRID_INF) ? group_max4 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max4 > -WAN_HYBRID_INF) ? group_max4 * softmax_scale_log2 : 0.0f)};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_0 + 64))[_lf], _fma_b2_16, _fma_c2_17);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    _tmem_load_0[_le + 64] = approx_exp2(_tmem_load_0[_le + 64]);
                }
                float2 _f2_13 = make_float2(_tmem_load_0[64], _tmem_load_0[65]);
                float2 partial_80 = _f2_13;
                #pragma unroll
                for (int pair_4 = 2; pair_4 < 16; pair_4 += 2) {
                    float2 _f2_14 = make_float2((_tmem_load_0 + 64)[pair_4], (_tmem_load_0 + 64)[pair_4 + 1]);
                    partial_80 = add_f32x2(partial_80, _f2_14);
                }
                float2 frag_sum2_81 = partial_80;
                float2 _f2_15 = make_float2(((group_max4 > -WAN_HYBRID_INF) ? _exp2_4 : 0.0f), ((group_max4 > -WAN_HYBRID_INF) ? _exp2_4 : 0.0f));
                float2 raw_scale2_82 = _f2_15;
                float2 raw_sum2_83 = mul_f32x2(frag_sum2_81, raw_scale2_82);
                uint32_t _fp4_4[2];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_4[0]) : "f"(_tmem_load_0[64]), "f"(_tmem_load_0[65]), "f"(_tmem_load_0[66]), "f"(_tmem_load_0[67]), "f"(_tmem_load_0[68]), "f"(_tmem_load_0[69]), "f"(_tmem_load_0[70]), "f"(_tmem_load_0[71]));
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_4[1]) : "f"(_tmem_load_0[72]), "f"(_tmem_load_0[73]), "f"(_tmem_load_0[74]), "f"(_tmem_load_0[75]), "f"(_tmem_load_0[76]), "f"(_tmem_load_0[77]), "f"(_tmem_load_0[78]), "f"(_tmem_load_0[79]));
                block_sum2 = add_f32x2(block_sum2, raw_sum2_83);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x2.b32"
                    " [%0], {%1, %2};"
                    :: "r"(taddr + (unsigned int)p_stage_off + 72 + (unsigned int)(warp % 4 * 32 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_4[1]))
                    : "memory");
                float _exp2_5 = approx_exp2(((group_max5 > -WAN_HYBRID_INF) ? group_max5 * softmax_scale_log2 : 0.0f) - new_max_scaled - 2.584962500721156f);
                sf_values[1] = ((group_max5 > -WAN_HYBRID_INF) ? _exp2_5 : 0.0f);
                const float2 _fma_b2_18 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_19 = {2.584962500721156f - ((group_max5 > -WAN_HYBRID_INF) ? group_max5 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max5 > -WAN_HYBRID_INF) ? group_max5 * softmax_scale_log2 : 0.0f)};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_0 + 80))[_lf], _fma_b2_18, _fma_c2_19);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    _tmem_load_0[_le + 80] = approx_exp2(_tmem_load_0[_le + 80]);
                }
                float2 _f2_16 = make_float2(_tmem_load_0[80], _tmem_load_0[81]);
                float2 partial_84 = _f2_16;
                #pragma unroll
                for (int pair_5 = 2; pair_5 < 16; pair_5 += 2) {
                    float2 _f2_17 = make_float2((_tmem_load_0 + 80)[pair_5], (_tmem_load_0 + 80)[pair_5 + 1]);
                    partial_84 = add_f32x2(partial_84, _f2_17);
                }
                float2 frag_sum2_85 = partial_84;
                float2 _f2_18 = make_float2(((group_max5 > -WAN_HYBRID_INF) ? _exp2_5 : 0.0f), ((group_max5 > -WAN_HYBRID_INF) ? _exp2_5 : 0.0f));
                float2 raw_scale2_86 = _f2_18;
                float2 raw_sum2_87 = mul_f32x2(frag_sum2_85, raw_scale2_86);
                uint32_t _fp4_5[2];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_5[0]) : "f"(_tmem_load_0[80]), "f"(_tmem_load_0[81]), "f"(_tmem_load_0[82]), "f"(_tmem_load_0[83]), "f"(_tmem_load_0[84]), "f"(_tmem_load_0[85]), "f"(_tmem_load_0[86]), "f"(_tmem_load_0[87]));
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_5[1]) : "f"(_tmem_load_0[88]), "f"(_tmem_load_0[89]), "f"(_tmem_load_0[90]), "f"(_tmem_load_0[91]), "f"(_tmem_load_0[92]), "f"(_tmem_load_0[93]), "f"(_tmem_load_0[94]), "f"(_tmem_load_0[95]));
                block_sum2 = add_f32x2(block_sum2, raw_sum2_87);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x2.b32"
                    " [%0], {%1, %2};"
                    :: "r"(taddr + (unsigned int)p_stage_off + 72 + (unsigned int)(warp % 4 * 32 << 16) + 2), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_5[1]))
                    : "memory");
                float _exp2_6 = approx_exp2(((group_max6 > -WAN_HYBRID_INF) ? group_max6 * softmax_scale_log2 : 0.0f) - new_max_scaled - 2.584962500721156f);
                sf_values[2] = ((group_max6 > -WAN_HYBRID_INF) ? _exp2_6 : 0.0f);
                const float2 _fma_b2_20 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_21 = {2.584962500721156f - ((group_max6 > -WAN_HYBRID_INF) ? group_max6 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max6 > -WAN_HYBRID_INF) ? group_max6 * softmax_scale_log2 : 0.0f)};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_0 + 96))[_lf], _fma_b2_20, _fma_c2_21);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    _tmem_load_0[_le + 96] = approx_exp2(_tmem_load_0[_le + 96]);
                }
                float2 _f2_19 = make_float2(_tmem_load_0[96], _tmem_load_0[97]);
                float2 partial_88 = _f2_19;
                #pragma unroll
                for (int pair_6 = 2; pair_6 < 16; pair_6 += 2) {
                    float2 _f2_20 = make_float2((_tmem_load_0 + 96)[pair_6], (_tmem_load_0 + 96)[pair_6 + 1]);
                    partial_88 = add_f32x2(partial_88, _f2_20);
                }
                float2 frag_sum2_89 = partial_88;
                float2 _f2_21 = make_float2(((group_max6 > -WAN_HYBRID_INF) ? _exp2_6 : 0.0f), ((group_max6 > -WAN_HYBRID_INF) ? _exp2_6 : 0.0f));
                float2 raw_scale2_90 = _f2_21;
                float2 raw_sum2_91 = mul_f32x2(frag_sum2_89, raw_scale2_90);
                uint32_t _fp4_6[2];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_6[0]) : "f"(_tmem_load_0[96]), "f"(_tmem_load_0[97]), "f"(_tmem_load_0[98]), "f"(_tmem_load_0[99]), "f"(_tmem_load_0[100]), "f"(_tmem_load_0[101]), "f"(_tmem_load_0[102]), "f"(_tmem_load_0[103]));
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_6[1]) : "f"(_tmem_load_0[104]), "f"(_tmem_load_0[105]), "f"(_tmem_load_0[106]), "f"(_tmem_load_0[107]), "f"(_tmem_load_0[108]), "f"(_tmem_load_0[109]), "f"(_tmem_load_0[110]), "f"(_tmem_load_0[111]));
                block_sum2 = add_f32x2(block_sum2, raw_sum2_91);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x2.b32"
                    " [%0], {%1, %2};"
                    :: "r"(taddr + (unsigned int)p_stage_off + 72 + (unsigned int)(warp % 4 * 32 << 16) + 4), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_6[1]))
                    : "memory");
                float _exp2_7 = approx_exp2(((group_max7 > -WAN_HYBRID_INF) ? group_max7 * softmax_scale_log2 : 0.0f) - new_max_scaled - 2.584962500721156f);
                sf_values[3] = ((group_max7 > -WAN_HYBRID_INF) ? _exp2_7 : 0.0f);
                const float2 _fma_b2_22 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_23 = {2.584962500721156f - ((group_max7 > -WAN_HYBRID_INF) ? group_max7 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max7 > -WAN_HYBRID_INF) ? group_max7 * softmax_scale_log2 : 0.0f)};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_0 + 112))[_lf], _fma_b2_22, _fma_c2_23);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    _tmem_load_0[_le + 112] = approx_exp2(_tmem_load_0[_le + 112]);
                }
                float2 _f2_22 = make_float2(_tmem_load_0[112], _tmem_load_0[113]);
                float2 partial_92 = _f2_22;
                #pragma unroll
                for (int pair_7 = 2; pair_7 < 16; pair_7 += 2) {
                    float2 _f2_23 = make_float2((_tmem_load_0 + 112)[pair_7], (_tmem_load_0 + 112)[pair_7 + 1]);
                    partial_92 = add_f32x2(partial_92, _f2_23);
                }
                float2 frag_sum2_93 = partial_92;
                float2 _f2_24 = make_float2(((group_max7 > -WAN_HYBRID_INF) ? _exp2_7 : 0.0f), ((group_max7 > -WAN_HYBRID_INF) ? _exp2_7 : 0.0f));
                float2 raw_scale2_94 = _f2_24;
                float2 raw_sum2_95 = mul_f32x2(frag_sum2_93, raw_scale2_94);
                uint32_t _fp4_7[2];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_7[0]) : "f"(_tmem_load_0[112]), "f"(_tmem_load_0[113]), "f"(_tmem_load_0[114]), "f"(_tmem_load_0[115]), "f"(_tmem_load_0[116]), "f"(_tmem_load_0[117]), "f"(_tmem_load_0[118]), "f"(_tmem_load_0[119]));
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_7[1]) : "f"(_tmem_load_0[120]), "f"(_tmem_load_0[121]), "f"(_tmem_load_0[122]), "f"(_tmem_load_0[123]), "f"(_tmem_load_0[124]), "f"(_tmem_load_0[125]), "f"(_tmem_load_0[126]), "f"(_tmem_load_0[127]));
                block_sum2 = add_f32x2(block_sum2, raw_sum2_95);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x2.b32"
                    " [%0], {%1, %2};"
                    :: "r"(taddr + (unsigned int)p_stage_off + 72 + (unsigned int)(warp % 4 * 32 << 16) + 6), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_7[1]))
                    : "memory");
                uint32_t _fp8_1[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(sf_values[0]), "f"(sf_values[1]),
                                           "f"(sf_values[2]), "f"(sf_values[3]));
                    _fp8_1[0] = _packed;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)p_stage_off + 84 + (unsigned int)(warp % 4)), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_1[0]))
                    : "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_2_addr + (stage) * 8);
                row_sum_val = block_sum2.x + block_sum2.y;
                #pragma unroll 1
                for (unsigned int n_iter = 1; n_iter < num_masked_iters; n_iter++) {
                    int n_block_0 = num_n_blocks - 1 - n_iter;
                    mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    int s_base_1 = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_1[128];
                    tmem_ld_x32(&_tmem_load_1[0], s_base_1);
                    tmem_ld_x32(&_tmem_load_1[32], s_base_1 + 32);
                    tmem_ld_x32(&_tmem_load_1[64], s_base_1 + 64);
                    tmem_ld_x32(&_tmem_load_1[96], s_base_1 + 96);
                    int valid_count = causal_row - n_block_0 * BLOCK_N + 1;
                    uint32_t _slice_lo_mask_8;
                    {
                        int _lim_24 = valid_count;
                        if (_lim_24 <= 0) { _slice_lo_mask_8 = 0u; }
                        else if (_lim_24 >= 32) { _slice_lo_mask_8 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_8) : "r"(_lim_24));
                        }
                    }
                    #pragma unroll
                    for (int _i_25 = 0; _i_25 < 32; _i_25++) {
                        if (!(_slice_lo_mask_8 & (1u << _i_25))) _tmem_load_1[0 + _i_25] = -WAN_HYBRID_INF;
                    }
                    uint32_t _slice_lo_mask_9;
                    {
                        int _lim_26 = valid_count - 32;
                        if (_lim_26 <= 0) { _slice_lo_mask_9 = 0u; }
                        else if (_lim_26 >= 32) { _slice_lo_mask_9 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_9) : "r"(_lim_26));
                        }
                    }
                    #pragma unroll
                    for (int _i_27 = 0; _i_27 < 32; _i_27++) {
                        if (!(_slice_lo_mask_9 & (1u << _i_27))) _tmem_load_1[32 + _i_27] = -WAN_HYBRID_INF;
                    }
                    uint32_t _slice_lo_mask_10;
                    {
                        int _lim_28 = valid_count - 64;
                        if (_lim_28 <= 0) { _slice_lo_mask_10 = 0u; }
                        else if (_lim_28 >= 32) { _slice_lo_mask_10 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_10) : "r"(_lim_28));
                        }
                    }
                    #pragma unroll
                    for (int _i_29 = 0; _i_29 < 32; _i_29++) {
                        if (!(_slice_lo_mask_10 & (1u << _i_29))) _tmem_load_1[64 + _i_29] = -WAN_HYBRID_INF;
                    }
                    uint32_t _slice_lo_mask_11;
                    {
                        int _lim_30 = valid_count - 96;
                        if (_lim_30 <= 0) { _slice_lo_mask_11 = 0u; }
                        else if (_lim_30 >= 32) { _slice_lo_mask_11 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_11) : "r"(_lim_30));
                        }
                    }
                    #pragma unroll
                    for (int _i_31 = 0; _i_31 < 32; _i_31++) {
                        if (!(_slice_lo_mask_11 & (1u << _i_31))) _tmem_load_1[96 + _i_31] = -WAN_HYBRID_INF;
                    }
                    float _max3_59;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_59) : "f"(_tmem_load_1[0]), "f"(_tmem_load_1[1]), "f"(_tmem_load_1[2]));
                    float max012_2 = _max3_59;
                    float _max3_60;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_60) : "f"(_tmem_load_1[3]), "f"(_tmem_load_1[4]), "f"(_tmem_load_1[5]));
                    float max345_3 = _max3_60;
                    float _max3_61;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_61) : "f"(_tmem_load_1[6]), "f"(_tmem_load_1[7]), "f"(_tmem_load_1[8]));
                    float max678_4 = _max3_61;
                    float _max3_62;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_62) : "f"(_tmem_load_1[9]), "f"(_tmem_load_1[10]), "f"(_tmem_load_1[11]));
                    float max9ab_5 = _max3_62;
                    float _max3_63;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_63) : "f"(_tmem_load_1[12]), "f"(_tmem_load_1[13]), "f"(_tmem_load_1[14]));
                    float maxcde_6 = _max3_63;
                    float _max3_64;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_64) : "f"(max012_2), "f"(max345_3), "f"(max678_4));
                    float max0_8_7 = _max3_64;
                    float _max3_65;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_65) : "f"(max9ab_5), "f"(maxcde_6), "f"(_tmem_load_1[15]));
                    float max9_f_8 = _max3_65;
                    float _max_9 = max_noftz(max0_8_7, max9_f_8);
                    float group_max0_9 = _max_9;
                    float _max3_66;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_66) : "f"(_tmem_load_1[16]), "f"(_tmem_load_1[17]), "f"(_tmem_load_1[18]));
                    float max012_10 = _max3_66;
                    float _max3_67;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_67) : "f"(_tmem_load_1[19]), "f"(_tmem_load_1[20]), "f"(_tmem_load_1[21]));
                    float max345_11 = _max3_67;
                    float _max3_68;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_68) : "f"(_tmem_load_1[22]), "f"(_tmem_load_1[23]), "f"(_tmem_load_1[24]));
                    float max678_12 = _max3_68;
                    float _max3_69;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_69) : "f"(_tmem_load_1[25]), "f"(_tmem_load_1[26]), "f"(_tmem_load_1[27]));
                    float max9ab_13 = _max3_69;
                    float _max3_70;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_70) : "f"(_tmem_load_1[28]), "f"(_tmem_load_1[29]), "f"(_tmem_load_1[30]));
                    float maxcde_14 = _max3_70;
                    float _max3_71;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_71) : "f"(max012_10), "f"(max345_11), "f"(max678_12));
                    float max0_8_15 = _max3_71;
                    float _max3_72;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_72) : "f"(max9ab_13), "f"(maxcde_14), "f"(_tmem_load_1[31]));
                    float max9_f_16 = _max3_72;
                    float _max_10 = max_noftz(max0_8_15, max9_f_16);
                    float group_max1_17 = _max_10;
                    float _max3_73;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_73) : "f"(_tmem_load_1[32]), "f"(_tmem_load_1[33]), "f"(_tmem_load_1[34]));
                    float max012_18 = _max3_73;
                    float _max3_74;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_74) : "f"(_tmem_load_1[35]), "f"(_tmem_load_1[36]), "f"(_tmem_load_1[37]));
                    float max345_19 = _max3_74;
                    float _max3_75;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_75) : "f"(_tmem_load_1[38]), "f"(_tmem_load_1[39]), "f"(_tmem_load_1[40]));
                    float max678_20 = _max3_75;
                    float _max3_76;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_76) : "f"(_tmem_load_1[41]), "f"(_tmem_load_1[42]), "f"(_tmem_load_1[43]));
                    float max9ab_21 = _max3_76;
                    float _max3_77;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_77) : "f"(_tmem_load_1[44]), "f"(_tmem_load_1[45]), "f"(_tmem_load_1[46]));
                    float maxcde_22 = _max3_77;
                    float _max3_78;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_78) : "f"(max012_18), "f"(max345_19), "f"(max678_20));
                    float max0_8_23 = _max3_78;
                    float _max3_79;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_79) : "f"(max9ab_21), "f"(maxcde_22), "f"(_tmem_load_1[47]));
                    float max9_f_24 = _max3_79;
                    float _max_11 = max_noftz(max0_8_23, max9_f_24);
                    float group_max2_25 = _max_11;
                    float _max3_80;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_80) : "f"(_tmem_load_1[48]), "f"(_tmem_load_1[49]), "f"(_tmem_load_1[50]));
                    float max012_26 = _max3_80;
                    float _max3_81;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_81) : "f"(_tmem_load_1[51]), "f"(_tmem_load_1[52]), "f"(_tmem_load_1[53]));
                    float max345_27 = _max3_81;
                    float _max3_82;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_82) : "f"(_tmem_load_1[54]), "f"(_tmem_load_1[55]), "f"(_tmem_load_1[56]));
                    float max678_28 = _max3_82;
                    float _max3_83;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_83) : "f"(_tmem_load_1[57]), "f"(_tmem_load_1[58]), "f"(_tmem_load_1[59]));
                    float max9ab_29 = _max3_83;
                    float _max3_84;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_84) : "f"(_tmem_load_1[60]), "f"(_tmem_load_1[61]), "f"(_tmem_load_1[62]));
                    float maxcde_30 = _max3_84;
                    float _max3_85;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_85) : "f"(max012_26), "f"(max345_27), "f"(max678_28));
                    float max0_8_31 = _max3_85;
                    float _max3_86;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_86) : "f"(max9ab_29), "f"(maxcde_30), "f"(_tmem_load_1[63]));
                    float max9_f_32 = _max3_86;
                    float _max_12 = max_noftz(max0_8_31, max9_f_32);
                    float group_max3_33 = _max_12;
                    float _max3_87;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_87) : "f"(_tmem_load_1[64]), "f"(_tmem_load_1[65]), "f"(_tmem_load_1[66]));
                    float max012_34 = _max3_87;
                    float _max3_88;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_88) : "f"(_tmem_load_1[67]), "f"(_tmem_load_1[68]), "f"(_tmem_load_1[69]));
                    float max345_35 = _max3_88;
                    float _max3_89;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_89) : "f"(_tmem_load_1[70]), "f"(_tmem_load_1[71]), "f"(_tmem_load_1[72]));
                    float max678_36 = _max3_89;
                    float _max3_90;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_90) : "f"(_tmem_load_1[73]), "f"(_tmem_load_1[74]), "f"(_tmem_load_1[75]));
                    float max9ab_37 = _max3_90;
                    float _max3_91;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_91) : "f"(_tmem_load_1[76]), "f"(_tmem_load_1[77]), "f"(_tmem_load_1[78]));
                    float maxcde_38 = _max3_91;
                    float _max3_92;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_92) : "f"(max012_34), "f"(max345_35), "f"(max678_36));
                    float max0_8_39 = _max3_92;
                    float _max3_93;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_93) : "f"(max9ab_37), "f"(maxcde_38), "f"(_tmem_load_1[79]));
                    float max9_f_40 = _max3_93;
                    float _max_13 = max_noftz(max0_8_39, max9_f_40);
                    float group_max4_41 = _max_13;
                    float _max3_94;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_94) : "f"(_tmem_load_1[80]), "f"(_tmem_load_1[81]), "f"(_tmem_load_1[82]));
                    float max012_42 = _max3_94;
                    float _max3_95;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_95) : "f"(_tmem_load_1[83]), "f"(_tmem_load_1[84]), "f"(_tmem_load_1[85]));
                    float max345_43 = _max3_95;
                    float _max3_96;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_96) : "f"(_tmem_load_1[86]), "f"(_tmem_load_1[87]), "f"(_tmem_load_1[88]));
                    float max678_44 = _max3_96;
                    float _max3_97;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_97) : "f"(_tmem_load_1[89]), "f"(_tmem_load_1[90]), "f"(_tmem_load_1[91]));
                    float max9ab_45 = _max3_97;
                    float _max3_98;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_98) : "f"(_tmem_load_1[92]), "f"(_tmem_load_1[93]), "f"(_tmem_load_1[94]));
                    float maxcde_46 = _max3_98;
                    float _max3_99;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_99) : "f"(max012_42), "f"(max345_43), "f"(max678_44));
                    float max0_8_47 = _max3_99;
                    float _max3_100;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_100) : "f"(max9ab_45), "f"(maxcde_46), "f"(_tmem_load_1[95]));
                    float max9_f_48 = _max3_100;
                    float _max_14 = max_noftz(max0_8_47, max9_f_48);
                    float group_max5_49 = _max_14;
                    float _max3_101;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_101) : "f"(_tmem_load_1[96]), "f"(_tmem_load_1[97]), "f"(_tmem_load_1[98]));
                    float max012_50 = _max3_101;
                    float _max3_102;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_102) : "f"(_tmem_load_1[99]), "f"(_tmem_load_1[100]), "f"(_tmem_load_1[101]));
                    float max345_51 = _max3_102;
                    float _max3_103;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_103) : "f"(_tmem_load_1[102]), "f"(_tmem_load_1[103]), "f"(_tmem_load_1[104]));
                    float max678_52 = _max3_103;
                    float _max3_104;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_104) : "f"(_tmem_load_1[105]), "f"(_tmem_load_1[106]), "f"(_tmem_load_1[107]));
                    float max9ab_53 = _max3_104;
                    float _max3_105;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_105) : "f"(_tmem_load_1[108]), "f"(_tmem_load_1[109]), "f"(_tmem_load_1[110]));
                    float maxcde_54 = _max3_105;
                    float _max3_106;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_106) : "f"(max012_50), "f"(max345_51), "f"(max678_52));
                    float max0_8_55 = _max3_106;
                    float _max3_107;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_107) : "f"(max9ab_53), "f"(maxcde_54), "f"(_tmem_load_1[111]));
                    float max9_f_56 = _max3_107;
                    float _max_15 = max_noftz(max0_8_55, max9_f_56);
                    float group_max6_57 = _max_15;
                    float _max3_108;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_108) : "f"(_tmem_load_1[112]), "f"(_tmem_load_1[113]), "f"(_tmem_load_1[114]));
                    float max012_58 = _max3_108;
                    float _max3_109;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_109) : "f"(_tmem_load_1[115]), "f"(_tmem_load_1[116]), "f"(_tmem_load_1[117]));
                    float max345_59 = _max3_109;
                    float _max3_110;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_110) : "f"(_tmem_load_1[118]), "f"(_tmem_load_1[119]), "f"(_tmem_load_1[120]));
                    float max678_60 = _max3_110;
                    float _max3_111;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_111) : "f"(_tmem_load_1[121]), "f"(_tmem_load_1[122]), "f"(_tmem_load_1[123]));
                    float max9ab_61 = _max3_111;
                    float _max3_112;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_112) : "f"(_tmem_load_1[124]), "f"(_tmem_load_1[125]), "f"(_tmem_load_1[126]));
                    float maxcde_62 = _max3_112;
                    float _max3_113;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_113) : "f"(max012_58), "f"(max345_59), "f"(max678_60));
                    float max0_8_63 = _max3_113;
                    float _max3_114;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_114) : "f"(max9ab_61), "f"(maxcde_62), "f"(_tmem_load_1[127]));
                    float max9_f_64 = _max3_114;
                    float _max_16 = max_noftz(max0_8_63, max9_f_64);
                    float group_max7_65 = _max_16;
                    float _max3_115;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_115) : "f"(group_max0_9), "f"(group_max1_17), "f"(group_max2_25));
                    float max012_66 = _max3_115;
                    float _max3_116;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_116) : "f"(group_max3_33), "f"(group_max4_41), "f"(group_max5_49));
                    float max345_67 = _max3_116;
                    float _max_17 = max_noftz(group_max6_57, group_max7_65);
                    float max67_68 = _max_17;
                    float _max3_117;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_117) : "f"(max012_66), "f"(max345_67), "f"(max67_68));
                    new_max = _max3_117;
                    group_max0 = group_max0_9;
                    group_max1 = group_max1_17;
                    group_max2 = group_max2_25;
                    group_max3 = group_max3_33;
                    group_max4 = group_max4_41;
                    group_max5 = group_max5_49;
                    group_max6 = group_max6_57;
                    group_max7 = group_max7_65;
                    float _max_18 = max_noftz(new_max, row_max_val);
                    new_max = _max_18;
                    float new_max_scaled_69 = ((new_max == -WAN_HYBRID_INF) ? 0.0f : new_max) * softmax_scale_log2;
                    float acc_scale;
                    float selected_max;
                    if ((row_max_val - ((new_max == -WAN_HYBRID_INF) ? 0.0f : new_max)) * softmax_scale_log2 >= -8.0f) {
                        selected_max = row_max_val;
                        acc_scale = 1.0f;
                        new_max_scaled_69 = ((row_max_val == -WAN_HYBRID_INF) ? 0.0f : row_max_val) * softmax_scale_log2;
                    } else {
                        selected_max = new_max;
                        if (row_max_val > -WAN_HYBRID_INF) {
                            float _exp2_8 = approx_exp2((row_max_val - ((new_max == -WAN_HYBRID_INF) ? 0.0f : new_max)) * softmax_scale_log2);
                            acc_scale = _exp2_8;
                        } else {
                            acc_scale = 1.0f;
                        }
                    }
                    row_max_val = selected_max;
                    sScale[warp % 4 * 32 + lane + scale_off] = acc_scale;
                    mbarrier_arrive(corr_sig_addr + (stage) * 8);
                    float2 _f2_25 = make_float2(0.0f, 0.0f);
                    float2 block_sum2_70 = _f2_25;
                    float sf_values_71[4];
                    int p_stage_off_72 = stage * 128;
                    float block_max_74 = group_max0;
                    float block_max_scaled_75 = ((block_max_74 > -WAN_HYBRID_INF) ? block_max_74 * softmax_scale_log2 : 0.0f);
                    float _exp2_9 = approx_exp2(block_max_scaled_75 - new_max_scaled_69 - 2.584962500721156f);
                    float p_scale_76 = ((block_max_74 > -WAN_HYBRID_INF) ? _exp2_9 : 0.0f);
                    sf_values_71[0] = p_scale_76;
                    const float2 _fma_b2_32 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_33 = {2.584962500721156f - block_max_scaled_75, 2.584962500721156f - block_max_scaled_75};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + 0))[_lf], _fma_b2_32, _fma_c2_33);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_1[_le] = approx_exp2(_tmem_load_1[_le]);
                    }
                    float2 _f2_26 = make_float2(_tmem_load_1[0], _tmem_load_1[1]);
                    float2 partial_77 = _f2_26;
                    #pragma unroll
                    for (int pair_8 = 2; pair_8 < 16; pair_8 += 2) {
                        float2 _f2_27 = make_float2((_tmem_load_1 + 0)[pair_8], (_tmem_load_1 + 0)[pair_8 + 1]);
                        partial_77 = add_f32x2(partial_77, _f2_27);
                    }
                    float2 frag_sum2_78 = partial_77;
                    float2 _f2_28 = make_float2(p_scale_76, p_scale_76);
                    float2 raw_scale2_79 = _f2_28;
                    float2 raw_sum2_80 = mul_f32x2(frag_sum2_78, raw_scale2_79);
                    uint32_t _fp4_8[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_8[0]) : "f"(_tmem_load_1[0]), "f"(_tmem_load_1[1]), "f"(_tmem_load_1[2]), "f"(_tmem_load_1[3]), "f"(_tmem_load_1[4]), "f"(_tmem_load_1[5]), "f"(_tmem_load_1[6]), "f"(_tmem_load_1[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_8[1]) : "f"(_tmem_load_1[8]), "f"(_tmem_load_1[9]), "f"(_tmem_load_1[10]), "f"(_tmem_load_1[11]), "f"(_tmem_load_1[12]), "f"(_tmem_load_1[13]), "f"(_tmem_load_1[14]), "f"(_tmem_load_1[15]));
                    block_sum2_70 = add_f32x2(block_sum2_70, raw_sum2_80);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72 + 64 + (unsigned int)(warp % 4 * 32 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_8[1]))
                        : "memory");
                    float block_max_81 = group_max1;
                    float block_max_scaled_82 = ((block_max_81 > -WAN_HYBRID_INF) ? block_max_81 * softmax_scale_log2 : 0.0f);
                    float _exp2_10 = approx_exp2(block_max_scaled_82 - new_max_scaled_69 - 2.584962500721156f);
                    float p_scale_83 = ((block_max_81 > -WAN_HYBRID_INF) ? _exp2_10 : 0.0f);
                    sf_values_71[1] = p_scale_83;
                    const float2 _fma_b2_34 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_35 = {2.584962500721156f - block_max_scaled_82, 2.584962500721156f - block_max_scaled_82};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + 16))[_lf], _fma_b2_34, _fma_c2_35);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_1[_le + 16] = approx_exp2(_tmem_load_1[_le + 16]);
                    }
                    float2 _f2_29 = make_float2(_tmem_load_1[16], _tmem_load_1[17]);
                    float2 partial_85 = _f2_29;
                    #pragma unroll
                    for (int pair_9 = 2; pair_9 < 16; pair_9 += 2) {
                        float2 _f2_30 = make_float2((_tmem_load_1 + 16)[pair_9], (_tmem_load_1 + 16)[pair_9 + 1]);
                        partial_85 = add_f32x2(partial_85, _f2_30);
                    }
                    float2 frag_sum2_86 = partial_85;
                    float2 _f2_31 = make_float2(p_scale_83, p_scale_83);
                    float2 raw_scale2_87 = _f2_31;
                    float2 raw_sum2_88 = mul_f32x2(frag_sum2_86, raw_scale2_87);
                    uint32_t _fp4_9[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_9[0]) : "f"(_tmem_load_1[16]), "f"(_tmem_load_1[17]), "f"(_tmem_load_1[18]), "f"(_tmem_load_1[19]), "f"(_tmem_load_1[20]), "f"(_tmem_load_1[21]), "f"(_tmem_load_1[22]), "f"(_tmem_load_1[23]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_9[1]) : "f"(_tmem_load_1[24]), "f"(_tmem_load_1[25]), "f"(_tmem_load_1[26]), "f"(_tmem_load_1[27]), "f"(_tmem_load_1[28]), "f"(_tmem_load_1[29]), "f"(_tmem_load_1[30]), "f"(_tmem_load_1[31]));
                    block_sum2_70 = add_f32x2(block_sum2_70, raw_sum2_88);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72 + 64 + (unsigned int)(warp % 4 * 32 << 16) + 2), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_9[1]))
                        : "memory");
                    float block_max_89 = group_max2;
                    float block_max_scaled_90 = ((block_max_89 > -WAN_HYBRID_INF) ? block_max_89 * softmax_scale_log2 : 0.0f);
                    float _exp2_11 = approx_exp2(block_max_scaled_90 - new_max_scaled_69 - 2.584962500721156f);
                    float p_scale_91 = ((block_max_89 > -WAN_HYBRID_INF) ? _exp2_11 : 0.0f);
                    sf_values_71[2] = p_scale_91;
                    const float2 _fma_b2_36 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_37 = {2.584962500721156f - block_max_scaled_90, 2.584962500721156f - block_max_scaled_90};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + 32))[_lf], _fma_b2_36, _fma_c2_37);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_1[_le + 32] = approx_exp2(_tmem_load_1[_le + 32]);
                    }
                    float2 _f2_32 = make_float2(_tmem_load_1[32], _tmem_load_1[33]);
                    float2 partial_93 = _f2_32;
                    #pragma unroll
                    for (int pair_10 = 2; pair_10 < 16; pair_10 += 2) {
                        float2 _f2_33 = make_float2((_tmem_load_1 + 32)[pair_10], (_tmem_load_1 + 32)[pair_10 + 1]);
                        partial_93 = add_f32x2(partial_93, _f2_33);
                    }
                    float2 frag_sum2_94 = partial_93;
                    float2 _f2_34 = make_float2(p_scale_91, p_scale_91);
                    float2 raw_scale2_95 = _f2_34;
                    float2 raw_sum2_96 = mul_f32x2(frag_sum2_94, raw_scale2_95);
                    uint32_t _fp4_10[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_10[0]) : "f"(_tmem_load_1[32]), "f"(_tmem_load_1[33]), "f"(_tmem_load_1[34]), "f"(_tmem_load_1[35]), "f"(_tmem_load_1[36]), "f"(_tmem_load_1[37]), "f"(_tmem_load_1[38]), "f"(_tmem_load_1[39]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_10[1]) : "f"(_tmem_load_1[40]), "f"(_tmem_load_1[41]), "f"(_tmem_load_1[42]), "f"(_tmem_load_1[43]), "f"(_tmem_load_1[44]), "f"(_tmem_load_1[45]), "f"(_tmem_load_1[46]), "f"(_tmem_load_1[47]));
                    block_sum2_70 = add_f32x2(block_sum2_70, raw_sum2_96);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72 + 64 + (unsigned int)(warp % 4 * 32 << 16) + 4), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_10[1]))
                        : "memory");
                    float block_max_97 = group_max3;
                    float block_max_scaled_98 = ((block_max_97 > -WAN_HYBRID_INF) ? block_max_97 * softmax_scale_log2 : 0.0f);
                    float _exp2_12 = approx_exp2(block_max_scaled_98 - new_max_scaled_69 - 2.584962500721156f);
                    float p_scale_99 = ((block_max_97 > -WAN_HYBRID_INF) ? _exp2_12 : 0.0f);
                    sf_values_71[3] = p_scale_99;
                    const float2 _fma_b2_38 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_39 = {2.584962500721156f - block_max_scaled_98, 2.584962500721156f - block_max_scaled_98};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + 48))[_lf], _fma_b2_38, _fma_c2_39);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_1[_le + 48] = approx_exp2(_tmem_load_1[_le + 48]);
                    }
                    float2 _f2_35 = make_float2(_tmem_load_1[48], _tmem_load_1[49]);
                    float2 partial_100 = _f2_35;
                    #pragma unroll
                    for (int pair_11 = 2; pair_11 < 16; pair_11 += 2) {
                        float2 _f2_36 = make_float2((_tmem_load_1 + 48)[pair_11], (_tmem_load_1 + 48)[pair_11 + 1]);
                        partial_100 = add_f32x2(partial_100, _f2_36);
                    }
                    float2 frag_sum2_101 = partial_100;
                    float2 _f2_37 = make_float2(p_scale_99, p_scale_99);
                    float2 raw_scale2_102 = _f2_37;
                    float2 raw_sum2_103 = mul_f32x2(frag_sum2_101, raw_scale2_102);
                    uint32_t _fp4_11[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_11[0]) : "f"(_tmem_load_1[48]), "f"(_tmem_load_1[49]), "f"(_tmem_load_1[50]), "f"(_tmem_load_1[51]), "f"(_tmem_load_1[52]), "f"(_tmem_load_1[53]), "f"(_tmem_load_1[54]), "f"(_tmem_load_1[55]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_11[1]) : "f"(_tmem_load_1[56]), "f"(_tmem_load_1[57]), "f"(_tmem_load_1[58]), "f"(_tmem_load_1[59]), "f"(_tmem_load_1[60]), "f"(_tmem_load_1[61]), "f"(_tmem_load_1[62]), "f"(_tmem_load_1[63]));
                    block_sum2_70 = add_f32x2(block_sum2_70, raw_sum2_103);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72 + 64 + (unsigned int)(warp % 4 * 32 << 16) + 6), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_11[1]))
                        : "memory");
                    uint32_t _fp8_2[1];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sf_values_71[0]), "f"(sf_values_71[1]),
                                               "f"(sf_values_71[2]), "f"(sf_values_71[3]));
                        _fp8_2[0] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72 + 80 + (unsigned int)(warp % 4)), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_2[0]))
                        : "memory");
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
                    float _exp2_13 = approx_exp2(((group_max4 > -WAN_HYBRID_INF) ? group_max4 * softmax_scale_log2 : 0.0f) - new_max_scaled_69 - 2.584962500721156f);
                    sf_values_71[0] = ((group_max4 > -WAN_HYBRID_INF) ? _exp2_13 : 0.0f);
                    const float2 _fma_b2_40 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_41 = {2.584962500721156f - ((group_max4 > -WAN_HYBRID_INF) ? group_max4 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max4 > -WAN_HYBRID_INF) ? group_max4 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + 64))[_lf], _fma_b2_40, _fma_c2_41);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_1[_le + 64] = approx_exp2(_tmem_load_1[_le + 64]);
                    }
                    float2 _f2_38 = make_float2(_tmem_load_1[64], _tmem_load_1[65]);
                    float2 partial_104 = _f2_38;
                    #pragma unroll
                    for (int pair_12 = 2; pair_12 < 16; pair_12 += 2) {
                        float2 _f2_39 = make_float2((_tmem_load_1 + 64)[pair_12], (_tmem_load_1 + 64)[pair_12 + 1]);
                        partial_104 = add_f32x2(partial_104, _f2_39);
                    }
                    float2 frag_sum2_105 = partial_104;
                    float2 _f2_40 = make_float2(((group_max4 > -WAN_HYBRID_INF) ? _exp2_13 : 0.0f), ((group_max4 > -WAN_HYBRID_INF) ? _exp2_13 : 0.0f));
                    float2 raw_scale2_106 = _f2_40;
                    float2 raw_sum2_107 = mul_f32x2(frag_sum2_105, raw_scale2_106);
                    uint32_t _fp4_12[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_12[0]) : "f"(_tmem_load_1[64]), "f"(_tmem_load_1[65]), "f"(_tmem_load_1[66]), "f"(_tmem_load_1[67]), "f"(_tmem_load_1[68]), "f"(_tmem_load_1[69]), "f"(_tmem_load_1[70]), "f"(_tmem_load_1[71]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_12[1]) : "f"(_tmem_load_1[72]), "f"(_tmem_load_1[73]), "f"(_tmem_load_1[74]), "f"(_tmem_load_1[75]), "f"(_tmem_load_1[76]), "f"(_tmem_load_1[77]), "f"(_tmem_load_1[78]), "f"(_tmem_load_1[79]));
                    block_sum2_70 = add_f32x2(block_sum2_70, raw_sum2_107);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72 + 72 + (unsigned int)(warp % 4 * 32 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_12[1]))
                        : "memory");
                    float _exp2_14 = approx_exp2(((group_max5 > -WAN_HYBRID_INF) ? group_max5 * softmax_scale_log2 : 0.0f) - new_max_scaled_69 - 2.584962500721156f);
                    sf_values_71[1] = ((group_max5 > -WAN_HYBRID_INF) ? _exp2_14 : 0.0f);
                    const float2 _fma_b2_42 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_43 = {2.584962500721156f - ((group_max5 > -WAN_HYBRID_INF) ? group_max5 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max5 > -WAN_HYBRID_INF) ? group_max5 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + 80))[_lf], _fma_b2_42, _fma_c2_43);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_1[_le + 80] = approx_exp2(_tmem_load_1[_le + 80]);
                    }
                    float2 _f2_41 = make_float2(_tmem_load_1[80], _tmem_load_1[81]);
                    float2 partial_108 = _f2_41;
                    #pragma unroll
                    for (int pair_13 = 2; pair_13 < 16; pair_13 += 2) {
                        float2 _f2_42 = make_float2((_tmem_load_1 + 80)[pair_13], (_tmem_load_1 + 80)[pair_13 + 1]);
                        partial_108 = add_f32x2(partial_108, _f2_42);
                    }
                    float2 frag_sum2_109 = partial_108;
                    float2 _f2_43 = make_float2(((group_max5 > -WAN_HYBRID_INF) ? _exp2_14 : 0.0f), ((group_max5 > -WAN_HYBRID_INF) ? _exp2_14 : 0.0f));
                    float2 raw_scale2_110 = _f2_43;
                    float2 raw_sum2_111 = mul_f32x2(frag_sum2_109, raw_scale2_110);
                    uint32_t _fp4_13[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_13[0]) : "f"(_tmem_load_1[80]), "f"(_tmem_load_1[81]), "f"(_tmem_load_1[82]), "f"(_tmem_load_1[83]), "f"(_tmem_load_1[84]), "f"(_tmem_load_1[85]), "f"(_tmem_load_1[86]), "f"(_tmem_load_1[87]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_13[1]) : "f"(_tmem_load_1[88]), "f"(_tmem_load_1[89]), "f"(_tmem_load_1[90]), "f"(_tmem_load_1[91]), "f"(_tmem_load_1[92]), "f"(_tmem_load_1[93]), "f"(_tmem_load_1[94]), "f"(_tmem_load_1[95]));
                    block_sum2_70 = add_f32x2(block_sum2_70, raw_sum2_111);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72 + 72 + (unsigned int)(warp % 4 * 32 << 16) + 2), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_13[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_13[1]))
                        : "memory");
                    float _exp2_15 = approx_exp2(((group_max6 > -WAN_HYBRID_INF) ? group_max6 * softmax_scale_log2 : 0.0f) - new_max_scaled_69 - 2.584962500721156f);
                    sf_values_71[2] = ((group_max6 > -WAN_HYBRID_INF) ? _exp2_15 : 0.0f);
                    const float2 _fma_b2_44 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_45 = {2.584962500721156f - ((group_max6 > -WAN_HYBRID_INF) ? group_max6 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max6 > -WAN_HYBRID_INF) ? group_max6 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + 96))[_lf], _fma_b2_44, _fma_c2_45);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_1[_le + 96] = approx_exp2(_tmem_load_1[_le + 96]);
                    }
                    float2 _f2_44 = make_float2(_tmem_load_1[96], _tmem_load_1[97]);
                    float2 partial_112 = _f2_44;
                    #pragma unroll
                    for (int pair_14 = 2; pair_14 < 16; pair_14 += 2) {
                        float2 _f2_45 = make_float2((_tmem_load_1 + 96)[pair_14], (_tmem_load_1 + 96)[pair_14 + 1]);
                        partial_112 = add_f32x2(partial_112, _f2_45);
                    }
                    float2 frag_sum2_113 = partial_112;
                    float2 _f2_46 = make_float2(((group_max6 > -WAN_HYBRID_INF) ? _exp2_15 : 0.0f), ((group_max6 > -WAN_HYBRID_INF) ? _exp2_15 : 0.0f));
                    float2 raw_scale2_114 = _f2_46;
                    float2 raw_sum2_115 = mul_f32x2(frag_sum2_113, raw_scale2_114);
                    uint32_t _fp4_14[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_14[0]) : "f"(_tmem_load_1[96]), "f"(_tmem_load_1[97]), "f"(_tmem_load_1[98]), "f"(_tmem_load_1[99]), "f"(_tmem_load_1[100]), "f"(_tmem_load_1[101]), "f"(_tmem_load_1[102]), "f"(_tmem_load_1[103]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_14[1]) : "f"(_tmem_load_1[104]), "f"(_tmem_load_1[105]), "f"(_tmem_load_1[106]), "f"(_tmem_load_1[107]), "f"(_tmem_load_1[108]), "f"(_tmem_load_1[109]), "f"(_tmem_load_1[110]), "f"(_tmem_load_1[111]));
                    block_sum2_70 = add_f32x2(block_sum2_70, raw_sum2_115);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72 + 72 + (unsigned int)(warp % 4 * 32 << 16) + 4), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_14[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_14[1]))
                        : "memory");
                    float _exp2_16 = approx_exp2(((group_max7 > -WAN_HYBRID_INF) ? group_max7 * softmax_scale_log2 : 0.0f) - new_max_scaled_69 - 2.584962500721156f);
                    sf_values_71[3] = ((group_max7 > -WAN_HYBRID_INF) ? _exp2_16 : 0.0f);
                    const float2 _fma_b2_46 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_47 = {2.584962500721156f - ((group_max7 > -WAN_HYBRID_INF) ? group_max7 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max7 > -WAN_HYBRID_INF) ? group_max7 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + 112))[_lf], _fma_b2_46, _fma_c2_47);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_1[_le + 112] = approx_exp2(_tmem_load_1[_le + 112]);
                    }
                    float2 _f2_47 = make_float2(_tmem_load_1[112], _tmem_load_1[113]);
                    float2 partial_116 = _f2_47;
                    #pragma unroll
                    for (int pair_15 = 2; pair_15 < 16; pair_15 += 2) {
                        float2 _f2_48 = make_float2((_tmem_load_1 + 112)[pair_15], (_tmem_load_1 + 112)[pair_15 + 1]);
                        partial_116 = add_f32x2(partial_116, _f2_48);
                    }
                    float2 frag_sum2_117 = partial_116;
                    float2 _f2_49 = make_float2(((group_max7 > -WAN_HYBRID_INF) ? _exp2_16 : 0.0f), ((group_max7 > -WAN_HYBRID_INF) ? _exp2_16 : 0.0f));
                    float2 raw_scale2_118 = _f2_49;
                    float2 raw_sum2_119 = mul_f32x2(frag_sum2_117, raw_scale2_118);
                    uint32_t _fp4_15[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_15[0]) : "f"(_tmem_load_1[112]), "f"(_tmem_load_1[113]), "f"(_tmem_load_1[114]), "f"(_tmem_load_1[115]), "f"(_tmem_load_1[116]), "f"(_tmem_load_1[117]), "f"(_tmem_load_1[118]), "f"(_tmem_load_1[119]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_15[1]) : "f"(_tmem_load_1[120]), "f"(_tmem_load_1[121]), "f"(_tmem_load_1[122]), "f"(_tmem_load_1[123]), "f"(_tmem_load_1[124]), "f"(_tmem_load_1[125]), "f"(_tmem_load_1[126]), "f"(_tmem_load_1[127]));
                    block_sum2_70 = add_f32x2(block_sum2_70, raw_sum2_119);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72 + 72 + (unsigned int)(warp % 4 * 32 << 16) + 6), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_15[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_15[1]))
                        : "memory");
                    uint32_t _fp8_3[1];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sf_values_71[0]), "f"(sf_values_71[1]),
                                               "f"(sf_values_71[2]), "f"(sf_values_71[3]));
                        _fp8_3[0] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72 + 84 + (unsigned int)(warp % 4)), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_3[0]))
                        : "memory");
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_2_addr + (stage) * 8);
                    row_sum_val = row_sum_val * acc_scale + (block_sum2_70.x + block_sum2_70.y);
                }
                unsigned int unmasked_begin = num_masked_iters;
                if (unmasked_begin < 1) {
                    unmasked_begin = 1;
                }
                #pragma unroll 1
                for (unsigned int n_iter_1 = unmasked_begin; n_iter_1 < num_n_blocks; n_iter_1++) {
                    mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    int s_base_0 = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_2[128];
                    tmem_ld_x32(&_tmem_load_2[0], s_base_0);
                    tmem_ld_x32(&_tmem_load_2[32], s_base_0 + 32);
                    tmem_ld_x32(&_tmem_load_2[64], s_base_0 + 64);
                    tmem_ld_x32(&_tmem_load_2[96], s_base_0 + 96);
                    float _max3_118;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_118) : "f"(_tmem_load_2[0]), "f"(_tmem_load_2[1]), "f"(_tmem_load_2[2]));
                    float max012_2_1 = _max3_118;
                    float _max3_119;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_119) : "f"(_tmem_load_2[3]), "f"(_tmem_load_2[4]), "f"(_tmem_load_2[5]));
                    float max345_3_1 = _max3_119;
                    float _max3_120;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_120) : "f"(_tmem_load_2[6]), "f"(_tmem_load_2[7]), "f"(_tmem_load_2[8]));
                    float max678_4_1 = _max3_120;
                    float _max3_121;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_121) : "f"(_tmem_load_2[9]), "f"(_tmem_load_2[10]), "f"(_tmem_load_2[11]));
                    float max9ab_5_1 = _max3_121;
                    float _max3_122;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_122) : "f"(_tmem_load_2[12]), "f"(_tmem_load_2[13]), "f"(_tmem_load_2[14]));
                    float maxcde_6_1 = _max3_122;
                    float _max3_123;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_123) : "f"(max012_2_1), "f"(max345_3_1), "f"(max678_4_1));
                    float max0_8_7_1 = _max3_123;
                    float _max3_124;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_124) : "f"(max9ab_5_1), "f"(maxcde_6_1), "f"(_tmem_load_2[15]));
                    float max9_f_8_1 = _max3_124;
                    float _max_19 = max_noftz(max0_8_7_1, max9_f_8_1);
                    float group_max0_9_1 = _max_19;
                    float _max3_125;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_125) : "f"(_tmem_load_2[16]), "f"(_tmem_load_2[17]), "f"(_tmem_load_2[18]));
                    float max012_10_1 = _max3_125;
                    float _max3_126;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_126) : "f"(_tmem_load_2[19]), "f"(_tmem_load_2[20]), "f"(_tmem_load_2[21]));
                    float max345_11_1 = _max3_126;
                    float _max3_127;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_127) : "f"(_tmem_load_2[22]), "f"(_tmem_load_2[23]), "f"(_tmem_load_2[24]));
                    float max678_12_1 = _max3_127;
                    float _max3_128;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_128) : "f"(_tmem_load_2[25]), "f"(_tmem_load_2[26]), "f"(_tmem_load_2[27]));
                    float max9ab_13_1 = _max3_128;
                    float _max3_129;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_129) : "f"(_tmem_load_2[28]), "f"(_tmem_load_2[29]), "f"(_tmem_load_2[30]));
                    float maxcde_14_1 = _max3_129;
                    float _max3_130;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_130) : "f"(max012_10_1), "f"(max345_11_1), "f"(max678_12_1));
                    float max0_8_15_1 = _max3_130;
                    float _max3_131;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_131) : "f"(max9ab_13_1), "f"(maxcde_14_1), "f"(_tmem_load_2[31]));
                    float max9_f_16_1 = _max3_131;
                    float _max_20 = max_noftz(max0_8_15_1, max9_f_16_1);
                    float group_max1_17_1 = _max_20;
                    float _max3_132;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_132) : "f"(_tmem_load_2[32]), "f"(_tmem_load_2[33]), "f"(_tmem_load_2[34]));
                    float max012_18_1 = _max3_132;
                    float _max3_133;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_133) : "f"(_tmem_load_2[35]), "f"(_tmem_load_2[36]), "f"(_tmem_load_2[37]));
                    float max345_19_1 = _max3_133;
                    float _max3_134;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_134) : "f"(_tmem_load_2[38]), "f"(_tmem_load_2[39]), "f"(_tmem_load_2[40]));
                    float max678_20_1 = _max3_134;
                    float _max3_135;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_135) : "f"(_tmem_load_2[41]), "f"(_tmem_load_2[42]), "f"(_tmem_load_2[43]));
                    float max9ab_21_1 = _max3_135;
                    float _max3_136;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_136) : "f"(_tmem_load_2[44]), "f"(_tmem_load_2[45]), "f"(_tmem_load_2[46]));
                    float maxcde_22_1 = _max3_136;
                    float _max3_137;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_137) : "f"(max012_18_1), "f"(max345_19_1), "f"(max678_20_1));
                    float max0_8_23_1 = _max3_137;
                    float _max3_138;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_138) : "f"(max9ab_21_1), "f"(maxcde_22_1), "f"(_tmem_load_2[47]));
                    float max9_f_24_1 = _max3_138;
                    float _max_21 = max_noftz(max0_8_23_1, max9_f_24_1);
                    float group_max2_25_1 = _max_21;
                    float _max3_139;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_139) : "f"(_tmem_load_2[48]), "f"(_tmem_load_2[49]), "f"(_tmem_load_2[50]));
                    float max012_26_1 = _max3_139;
                    float _max3_140;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_140) : "f"(_tmem_load_2[51]), "f"(_tmem_load_2[52]), "f"(_tmem_load_2[53]));
                    float max345_27_1 = _max3_140;
                    float _max3_141;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_141) : "f"(_tmem_load_2[54]), "f"(_tmem_load_2[55]), "f"(_tmem_load_2[56]));
                    float max678_28_1 = _max3_141;
                    float _max3_142;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_142) : "f"(_tmem_load_2[57]), "f"(_tmem_load_2[58]), "f"(_tmem_load_2[59]));
                    float max9ab_29_1 = _max3_142;
                    float _max3_143;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_143) : "f"(_tmem_load_2[60]), "f"(_tmem_load_2[61]), "f"(_tmem_load_2[62]));
                    float maxcde_30_1 = _max3_143;
                    float _max3_144;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_144) : "f"(max012_26_1), "f"(max345_27_1), "f"(max678_28_1));
                    float max0_8_31_1 = _max3_144;
                    float _max3_145;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_145) : "f"(max9ab_29_1), "f"(maxcde_30_1), "f"(_tmem_load_2[63]));
                    float max9_f_32_1 = _max3_145;
                    float _max_22 = max_noftz(max0_8_31_1, max9_f_32_1);
                    float group_max3_33_1 = _max_22;
                    float _max3_146;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_146) : "f"(_tmem_load_2[64]), "f"(_tmem_load_2[65]), "f"(_tmem_load_2[66]));
                    float max012_34_1 = _max3_146;
                    float _max3_147;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_147) : "f"(_tmem_load_2[67]), "f"(_tmem_load_2[68]), "f"(_tmem_load_2[69]));
                    float max345_35_1 = _max3_147;
                    float _max3_148;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_148) : "f"(_tmem_load_2[70]), "f"(_tmem_load_2[71]), "f"(_tmem_load_2[72]));
                    float max678_36_1 = _max3_148;
                    float _max3_149;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_149) : "f"(_tmem_load_2[73]), "f"(_tmem_load_2[74]), "f"(_tmem_load_2[75]));
                    float max9ab_37_1 = _max3_149;
                    float _max3_150;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_150) : "f"(_tmem_load_2[76]), "f"(_tmem_load_2[77]), "f"(_tmem_load_2[78]));
                    float maxcde_38_1 = _max3_150;
                    float _max3_151;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_151) : "f"(max012_34_1), "f"(max345_35_1), "f"(max678_36_1));
                    float max0_8_39_1 = _max3_151;
                    float _max3_152;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_152) : "f"(max9ab_37_1), "f"(maxcde_38_1), "f"(_tmem_load_2[79]));
                    float max9_f_40_1 = _max3_152;
                    float _max_23 = max_noftz(max0_8_39_1, max9_f_40_1);
                    float group_max4_41_1 = _max_23;
                    float _max3_153;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_153) : "f"(_tmem_load_2[80]), "f"(_tmem_load_2[81]), "f"(_tmem_load_2[82]));
                    float max012_42_1 = _max3_153;
                    float _max3_154;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_154) : "f"(_tmem_load_2[83]), "f"(_tmem_load_2[84]), "f"(_tmem_load_2[85]));
                    float max345_43_1 = _max3_154;
                    float _max3_155;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_155) : "f"(_tmem_load_2[86]), "f"(_tmem_load_2[87]), "f"(_tmem_load_2[88]));
                    float max678_44_1 = _max3_155;
                    float _max3_156;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_156) : "f"(_tmem_load_2[89]), "f"(_tmem_load_2[90]), "f"(_tmem_load_2[91]));
                    float max9ab_45_1 = _max3_156;
                    float _max3_157;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_157) : "f"(_tmem_load_2[92]), "f"(_tmem_load_2[93]), "f"(_tmem_load_2[94]));
                    float maxcde_46_1 = _max3_157;
                    float _max3_158;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_158) : "f"(max012_42_1), "f"(max345_43_1), "f"(max678_44_1));
                    float max0_8_47_1 = _max3_158;
                    float _max3_159;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_159) : "f"(max9ab_45_1), "f"(maxcde_46_1), "f"(_tmem_load_2[95]));
                    float max9_f_48_1 = _max3_159;
                    float _max_24 = max_noftz(max0_8_47_1, max9_f_48_1);
                    float group_max5_49_1 = _max_24;
                    float _max3_160;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_160) : "f"(_tmem_load_2[96]), "f"(_tmem_load_2[97]), "f"(_tmem_load_2[98]));
                    float max012_50_1 = _max3_160;
                    float _max3_161;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_161) : "f"(_tmem_load_2[99]), "f"(_tmem_load_2[100]), "f"(_tmem_load_2[101]));
                    float max345_51_1 = _max3_161;
                    float _max3_162;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_162) : "f"(_tmem_load_2[102]), "f"(_tmem_load_2[103]), "f"(_tmem_load_2[104]));
                    float max678_52_1 = _max3_162;
                    float _max3_163;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_163) : "f"(_tmem_load_2[105]), "f"(_tmem_load_2[106]), "f"(_tmem_load_2[107]));
                    float max9ab_53_1 = _max3_163;
                    float _max3_164;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_164) : "f"(_tmem_load_2[108]), "f"(_tmem_load_2[109]), "f"(_tmem_load_2[110]));
                    float maxcde_54_1 = _max3_164;
                    float _max3_165;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_165) : "f"(max012_50_1), "f"(max345_51_1), "f"(max678_52_1));
                    float max0_8_55_1 = _max3_165;
                    float _max3_166;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_166) : "f"(max9ab_53_1), "f"(maxcde_54_1), "f"(_tmem_load_2[111]));
                    float max9_f_56_1 = _max3_166;
                    float _max_25 = max_noftz(max0_8_55_1, max9_f_56_1);
                    float group_max6_57_1 = _max_25;
                    float _max3_167;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_167) : "f"(_tmem_load_2[112]), "f"(_tmem_load_2[113]), "f"(_tmem_load_2[114]));
                    float max012_58_1 = _max3_167;
                    float _max3_168;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_168) : "f"(_tmem_load_2[115]), "f"(_tmem_load_2[116]), "f"(_tmem_load_2[117]));
                    float max345_59_1 = _max3_168;
                    float _max3_169;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_169) : "f"(_tmem_load_2[118]), "f"(_tmem_load_2[119]), "f"(_tmem_load_2[120]));
                    float max678_60_1 = _max3_169;
                    float _max3_170;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_170) : "f"(_tmem_load_2[121]), "f"(_tmem_load_2[122]), "f"(_tmem_load_2[123]));
                    float max9ab_61_1 = _max3_170;
                    float _max3_171;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_171) : "f"(_tmem_load_2[124]), "f"(_tmem_load_2[125]), "f"(_tmem_load_2[126]));
                    float maxcde_62_1 = _max3_171;
                    float _max3_172;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_172) : "f"(max012_58_1), "f"(max345_59_1), "f"(max678_60_1));
                    float max0_8_63_1 = _max3_172;
                    float _max3_173;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_173) : "f"(max9ab_61_1), "f"(maxcde_62_1), "f"(_tmem_load_2[127]));
                    float max9_f_64_1 = _max3_173;
                    float _max_26 = max_noftz(max0_8_63_1, max9_f_64_1);
                    float group_max7_65_1 = _max_26;
                    float _max3_174;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_174) : "f"(group_max0_9_1), "f"(group_max1_17_1), "f"(group_max2_25_1));
                    float max012_66_1 = _max3_174;
                    float _max3_175;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_175) : "f"(group_max3_33_1), "f"(group_max4_41_1), "f"(group_max5_49_1));
                    float max345_67_1 = _max3_175;
                    float _max_27 = max_noftz(group_max6_57_1, group_max7_65_1);
                    float max67_68_1 = _max_27;
                    float _max3_176;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                    #endif
                    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_176) : "f"(max012_66_1), "f"(max345_67_1), "f"(max67_68_1));
                    new_max = _max3_176;
                    group_max0 = group_max0_9_1;
                    group_max1 = group_max1_17_1;
                    group_max2 = group_max2_25_1;
                    group_max3 = group_max3_33_1;
                    group_max4 = group_max4_41_1;
                    group_max5 = group_max5_49_1;
                    group_max6 = group_max6_57_1;
                    group_max7 = group_max7_65_1;
                    float _max_28 = max_noftz(new_max, row_max_val);
                    new_max = _max_28;
                    float new_max_scaled_69_1 = ((new_max == -WAN_HYBRID_INF) ? 0.0f : new_max) * softmax_scale_log2;
                    float acc_scale_1;
                    float selected_max_1;
                    selected_max_1 = (((row_max_val - ((new_max == -WAN_HYBRID_INF) ? 0.0f : new_max)) * softmax_scale_log2 >= -8.0f) ? row_max_val : new_max);
                    float _exp2_17 = approx_exp2((row_max_val - ((new_max == -WAN_HYBRID_INF) ? 0.0f : new_max)) * softmax_scale_log2);
                    acc_scale_1 = (((row_max_val - ((new_max == -WAN_HYBRID_INF) ? 0.0f : new_max)) * softmax_scale_log2 >= -8.0f) ? 1.0f : _exp2_17);
                    new_max_scaled_69_1 = (((row_max_val - ((new_max == -WAN_HYBRID_INF) ? 0.0f : new_max)) * softmax_scale_log2 >= -8.0f) ? row_max_val : ((new_max == -WAN_HYBRID_INF) ? 0.0f : new_max)) * softmax_scale_log2;
                    row_max_val = selected_max_1;
                    sScale[warp % 4 * 32 + lane + scale_off] = acc_scale_1;
                    mbarrier_arrive(corr_sig_addr + (stage) * 8);
                    float2 _f2_50 = make_float2(0.0f, 0.0f);
                    float2 block_sum2_70_1 = _f2_50;
                    float sf_values_71_1[4];
                    int p_stage_off_72_1 = stage * 128;
                    float block_max_74_1 = group_max0;
                    float block_max_scaled_75_1 = ((block_max_74_1 > -WAN_HYBRID_INF) ? block_max_74_1 * softmax_scale_log2 : 0.0f);
                    float _exp2_18 = approx_exp2(block_max_scaled_75_1 - new_max_scaled_69_1 - 2.584962500721156f);
                    float p_scale_76_1 = ((block_max_74_1 > -WAN_HYBRID_INF) ? _exp2_18 : 0.0f);
                    sf_values_71_1[0] = p_scale_76_1;
                    const float2 _fma_b2_48 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_49 = {2.584962500721156f - block_max_scaled_75_1, 2.584962500721156f - block_max_scaled_75_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_2 + 0))[_lf], _fma_b2_48, _fma_c2_49);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_2[_le] = approx_exp2(_tmem_load_2[_le]);
                    }
                    float2 _f2_51 = make_float2(_tmem_load_2[0], _tmem_load_2[1]);
                    float2 partial_77_1 = _f2_51;
                    #pragma unroll
                    for (int pair_16 = 2; pair_16 < 16; pair_16 += 2) {
                        float2 _f2_52 = make_float2((_tmem_load_2 + 0)[pair_16], (_tmem_load_2 + 0)[pair_16 + 1]);
                        partial_77_1 = add_f32x2(partial_77_1, _f2_52);
                    }
                    float2 frag_sum2_78_1 = partial_77_1;
                    float2 _f2_53 = make_float2(p_scale_76_1, p_scale_76_1);
                    float2 raw_scale2_79_1 = _f2_53;
                    float2 raw_sum2_80_1 = mul_f32x2(frag_sum2_78_1, raw_scale2_79_1);
                    uint32_t _fp4_16[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_16[0]) : "f"(_tmem_load_2[0]), "f"(_tmem_load_2[1]), "f"(_tmem_load_2[2]), "f"(_tmem_load_2[3]), "f"(_tmem_load_2[4]), "f"(_tmem_load_2[5]), "f"(_tmem_load_2[6]), "f"(_tmem_load_2[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_16[1]) : "f"(_tmem_load_2[8]), "f"(_tmem_load_2[9]), "f"(_tmem_load_2[10]), "f"(_tmem_load_2[11]), "f"(_tmem_load_2[12]), "f"(_tmem_load_2[13]), "f"(_tmem_load_2[14]), "f"(_tmem_load_2[15]));
                    block_sum2_70_1 = add_f32x2(block_sum2_70_1, raw_sum2_80_1);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72_1 + 64 + (unsigned int)(warp % 4 * 32 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_16[1]))
                        : "memory");
                    float block_max_81_1 = group_max1;
                    float block_max_scaled_82_1 = ((block_max_81_1 > -WAN_HYBRID_INF) ? block_max_81_1 * softmax_scale_log2 : 0.0f);
                    float _exp2_19 = approx_exp2(block_max_scaled_82_1 - new_max_scaled_69_1 - 2.584962500721156f);
                    float p_scale_83_1 = ((block_max_81_1 > -WAN_HYBRID_INF) ? _exp2_19 : 0.0f);
                    sf_values_71_1[1] = p_scale_83_1;
                    const float2 _fma_b2_50 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_51 = {2.584962500721156f - block_max_scaled_82_1, 2.584962500721156f - block_max_scaled_82_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_2 + 16))[_lf], _fma_b2_50, _fma_c2_51);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_2[_le + 16] = approx_exp2(_tmem_load_2[_le + 16]);
                    }
                    float2 _f2_54 = make_float2(_tmem_load_2[16], _tmem_load_2[17]);
                    float2 partial_85_1 = _f2_54;
                    #pragma unroll
                    for (int pair_17 = 2; pair_17 < 16; pair_17 += 2) {
                        float2 _f2_55 = make_float2((_tmem_load_2 + 16)[pair_17], (_tmem_load_2 + 16)[pair_17 + 1]);
                        partial_85_1 = add_f32x2(partial_85_1, _f2_55);
                    }
                    float2 frag_sum2_86_1 = partial_85_1;
                    float2 _f2_56 = make_float2(p_scale_83_1, p_scale_83_1);
                    float2 raw_scale2_87_1 = _f2_56;
                    float2 raw_sum2_88_1 = mul_f32x2(frag_sum2_86_1, raw_scale2_87_1);
                    uint32_t _fp4_17[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_17[0]) : "f"(_tmem_load_2[16]), "f"(_tmem_load_2[17]), "f"(_tmem_load_2[18]), "f"(_tmem_load_2[19]), "f"(_tmem_load_2[20]), "f"(_tmem_load_2[21]), "f"(_tmem_load_2[22]), "f"(_tmem_load_2[23]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_17[1]) : "f"(_tmem_load_2[24]), "f"(_tmem_load_2[25]), "f"(_tmem_load_2[26]), "f"(_tmem_load_2[27]), "f"(_tmem_load_2[28]), "f"(_tmem_load_2[29]), "f"(_tmem_load_2[30]), "f"(_tmem_load_2[31]));
                    block_sum2_70_1 = add_f32x2(block_sum2_70_1, raw_sum2_88_1);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72_1 + 64 + (unsigned int)(warp % 4 * 32 << 16) + 2), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_17[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_17[1]))
                        : "memory");
                    float block_max_89_1 = group_max2;
                    float block_max_scaled_90_1 = ((block_max_89_1 > -WAN_HYBRID_INF) ? block_max_89_1 * softmax_scale_log2 : 0.0f);
                    float _exp2_20 = approx_exp2(block_max_scaled_90_1 - new_max_scaled_69_1 - 2.584962500721156f);
                    float p_scale_91_1 = ((block_max_89_1 > -WAN_HYBRID_INF) ? _exp2_20 : 0.0f);
                    sf_values_71_1[2] = p_scale_91_1;
                    const float2 _fma_b2_52 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_53 = {2.584962500721156f - block_max_scaled_90_1, 2.584962500721156f - block_max_scaled_90_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_2 + 32))[_lf], _fma_b2_52, _fma_c2_53);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_2[_le + 32] = approx_exp2(_tmem_load_2[_le + 32]);
                    }
                    float2 _f2_57 = make_float2(_tmem_load_2[32], _tmem_load_2[33]);
                    float2 partial_93_1 = _f2_57;
                    #pragma unroll
                    for (int pair_18 = 2; pair_18 < 16; pair_18 += 2) {
                        float2 _f2_58 = make_float2((_tmem_load_2 + 32)[pair_18], (_tmem_load_2 + 32)[pair_18 + 1]);
                        partial_93_1 = add_f32x2(partial_93_1, _f2_58);
                    }
                    float2 frag_sum2_94_1 = partial_93_1;
                    float2 _f2_59 = make_float2(p_scale_91_1, p_scale_91_1);
                    float2 raw_scale2_95_1 = _f2_59;
                    float2 raw_sum2_96_1 = mul_f32x2(frag_sum2_94_1, raw_scale2_95_1);
                    uint32_t _fp4_18[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_18[0]) : "f"(_tmem_load_2[32]), "f"(_tmem_load_2[33]), "f"(_tmem_load_2[34]), "f"(_tmem_load_2[35]), "f"(_tmem_load_2[36]), "f"(_tmem_load_2[37]), "f"(_tmem_load_2[38]), "f"(_tmem_load_2[39]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_18[1]) : "f"(_tmem_load_2[40]), "f"(_tmem_load_2[41]), "f"(_tmem_load_2[42]), "f"(_tmem_load_2[43]), "f"(_tmem_load_2[44]), "f"(_tmem_load_2[45]), "f"(_tmem_load_2[46]), "f"(_tmem_load_2[47]));
                    block_sum2_70_1 = add_f32x2(block_sum2_70_1, raw_sum2_96_1);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72_1 + 64 + (unsigned int)(warp % 4 * 32 << 16) + 4), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_18[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_18[1]))
                        : "memory");
                    float block_max_97_1 = group_max3;
                    float block_max_scaled_98_1 = ((block_max_97_1 > -WAN_HYBRID_INF) ? block_max_97_1 * softmax_scale_log2 : 0.0f);
                    float _exp2_21 = approx_exp2(block_max_scaled_98_1 - new_max_scaled_69_1 - 2.584962500721156f);
                    float p_scale_99_1 = ((block_max_97_1 > -WAN_HYBRID_INF) ? _exp2_21 : 0.0f);
                    sf_values_71_1[3] = p_scale_99_1;
                    const float2 _fma_b2_54 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_55 = {2.584962500721156f - block_max_scaled_98_1, 2.584962500721156f - block_max_scaled_98_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_2 + 48))[_lf], _fma_b2_54, _fma_c2_55);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_2[_le + 48] = approx_exp2(_tmem_load_2[_le + 48]);
                    }
                    float2 _f2_60 = make_float2(_tmem_load_2[48], _tmem_load_2[49]);
                    float2 partial_100_1 = _f2_60;
                    #pragma unroll
                    for (int pair_19 = 2; pair_19 < 16; pair_19 += 2) {
                        float2 _f2_61 = make_float2((_tmem_load_2 + 48)[pair_19], (_tmem_load_2 + 48)[pair_19 + 1]);
                        partial_100_1 = add_f32x2(partial_100_1, _f2_61);
                    }
                    float2 frag_sum2_101_1 = partial_100_1;
                    float2 _f2_62 = make_float2(p_scale_99_1, p_scale_99_1);
                    float2 raw_scale2_102_1 = _f2_62;
                    float2 raw_sum2_103_1 = mul_f32x2(frag_sum2_101_1, raw_scale2_102_1);
                    uint32_t _fp4_19[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_19[0]) : "f"(_tmem_load_2[48]), "f"(_tmem_load_2[49]), "f"(_tmem_load_2[50]), "f"(_tmem_load_2[51]), "f"(_tmem_load_2[52]), "f"(_tmem_load_2[53]), "f"(_tmem_load_2[54]), "f"(_tmem_load_2[55]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_19[1]) : "f"(_tmem_load_2[56]), "f"(_tmem_load_2[57]), "f"(_tmem_load_2[58]), "f"(_tmem_load_2[59]), "f"(_tmem_load_2[60]), "f"(_tmem_load_2[61]), "f"(_tmem_load_2[62]), "f"(_tmem_load_2[63]));
                    block_sum2_70_1 = add_f32x2(block_sum2_70_1, raw_sum2_103_1);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72_1 + 64 + (unsigned int)(warp % 4 * 32 << 16) + 6), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_19[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_19[1]))
                        : "memory");
                    uint32_t _fp8_4[1];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sf_values_71_1[0]), "f"(sf_values_71_1[1]),
                                               "f"(sf_values_71_1[2]), "f"(sf_values_71_1[3]));
                        _fp8_4[0] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72_1 + 80 + (unsigned int)(warp % 4)), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_4[0]))
                        : "memory");
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + (stage) * 8);
                    float _exp2_22 = approx_exp2(((group_max4 > -WAN_HYBRID_INF) ? group_max4 * softmax_scale_log2 : 0.0f) - new_max_scaled_69_1 - 2.584962500721156f);
                    sf_values_71_1[0] = ((group_max4 > -WAN_HYBRID_INF) ? _exp2_22 : 0.0f);
                    const float2 _fma_b2_56 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_57 = {2.584962500721156f - ((group_max4 > -WAN_HYBRID_INF) ? group_max4 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max4 > -WAN_HYBRID_INF) ? group_max4 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_2 + 64))[_lf], _fma_b2_56, _fma_c2_57);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_2[_le + 64] = approx_exp2(_tmem_load_2[_le + 64]);
                    }
                    float2 _f2_63 = make_float2(_tmem_load_2[64], _tmem_load_2[65]);
                    float2 partial_104_1 = _f2_63;
                    #pragma unroll
                    for (int pair_20 = 2; pair_20 < 16; pair_20 += 2) {
                        float2 _f2_64 = make_float2((_tmem_load_2 + 64)[pair_20], (_tmem_load_2 + 64)[pair_20 + 1]);
                        partial_104_1 = add_f32x2(partial_104_1, _f2_64);
                    }
                    float2 frag_sum2_105_1 = partial_104_1;
                    float2 _f2_65 = make_float2(((group_max4 > -WAN_HYBRID_INF) ? _exp2_22 : 0.0f), ((group_max4 > -WAN_HYBRID_INF) ? _exp2_22 : 0.0f));
                    float2 raw_scale2_106_1 = _f2_65;
                    float2 raw_sum2_107_1 = mul_f32x2(frag_sum2_105_1, raw_scale2_106_1);
                    uint32_t _fp4_20[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_20[0]) : "f"(_tmem_load_2[64]), "f"(_tmem_load_2[65]), "f"(_tmem_load_2[66]), "f"(_tmem_load_2[67]), "f"(_tmem_load_2[68]), "f"(_tmem_load_2[69]), "f"(_tmem_load_2[70]), "f"(_tmem_load_2[71]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_20[1]) : "f"(_tmem_load_2[72]), "f"(_tmem_load_2[73]), "f"(_tmem_load_2[74]), "f"(_tmem_load_2[75]), "f"(_tmem_load_2[76]), "f"(_tmem_load_2[77]), "f"(_tmem_load_2[78]), "f"(_tmem_load_2[79]));
                    block_sum2_70_1 = add_f32x2(block_sum2_70_1, raw_sum2_107_1);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72_1 + 72 + (unsigned int)(warp % 4 * 32 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_20[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_20[1]))
                        : "memory");
                    float _exp2_23 = approx_exp2(((group_max5 > -WAN_HYBRID_INF) ? group_max5 * softmax_scale_log2 : 0.0f) - new_max_scaled_69_1 - 2.584962500721156f);
                    sf_values_71_1[1] = ((group_max5 > -WAN_HYBRID_INF) ? _exp2_23 : 0.0f);
                    const float2 _fma_b2_58 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_59 = {2.584962500721156f - ((group_max5 > -WAN_HYBRID_INF) ? group_max5 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max5 > -WAN_HYBRID_INF) ? group_max5 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_2 + 80))[_lf], _fma_b2_58, _fma_c2_59);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_2[_le + 80] = approx_exp2(_tmem_load_2[_le + 80]);
                    }
                    float2 _f2_66 = make_float2(_tmem_load_2[80], _tmem_load_2[81]);
                    float2 partial_108_1 = _f2_66;
                    #pragma unroll
                    for (int pair_21 = 2; pair_21 < 16; pair_21 += 2) {
                        float2 _f2_67 = make_float2((_tmem_load_2 + 80)[pair_21], (_tmem_load_2 + 80)[pair_21 + 1]);
                        partial_108_1 = add_f32x2(partial_108_1, _f2_67);
                    }
                    float2 frag_sum2_109_1 = partial_108_1;
                    float2 _f2_68 = make_float2(((group_max5 > -WAN_HYBRID_INF) ? _exp2_23 : 0.0f), ((group_max5 > -WAN_HYBRID_INF) ? _exp2_23 : 0.0f));
                    float2 raw_scale2_110_1 = _f2_68;
                    float2 raw_sum2_111_1 = mul_f32x2(frag_sum2_109_1, raw_scale2_110_1);
                    uint32_t _fp4_21[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_21[0]) : "f"(_tmem_load_2[80]), "f"(_tmem_load_2[81]), "f"(_tmem_load_2[82]), "f"(_tmem_load_2[83]), "f"(_tmem_load_2[84]), "f"(_tmem_load_2[85]), "f"(_tmem_load_2[86]), "f"(_tmem_load_2[87]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_21[1]) : "f"(_tmem_load_2[88]), "f"(_tmem_load_2[89]), "f"(_tmem_load_2[90]), "f"(_tmem_load_2[91]), "f"(_tmem_load_2[92]), "f"(_tmem_load_2[93]), "f"(_tmem_load_2[94]), "f"(_tmem_load_2[95]));
                    block_sum2_70_1 = add_f32x2(block_sum2_70_1, raw_sum2_111_1);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72_1 + 72 + (unsigned int)(warp % 4 * 32 << 16) + 2), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_21[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_21[1]))
                        : "memory");
                    float _exp2_24 = approx_exp2(((group_max6 > -WAN_HYBRID_INF) ? group_max6 * softmax_scale_log2 : 0.0f) - new_max_scaled_69_1 - 2.584962500721156f);
                    sf_values_71_1[2] = ((group_max6 > -WAN_HYBRID_INF) ? _exp2_24 : 0.0f);
                    const float2 _fma_b2_60 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_61 = {2.584962500721156f - ((group_max6 > -WAN_HYBRID_INF) ? group_max6 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max6 > -WAN_HYBRID_INF) ? group_max6 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_2 + 96))[_lf], _fma_b2_60, _fma_c2_61);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_2[_le + 96] = approx_exp2(_tmem_load_2[_le + 96]);
                    }
                    float2 _f2_69 = make_float2(_tmem_load_2[96], _tmem_load_2[97]);
                    float2 partial_112_1 = _f2_69;
                    #pragma unroll
                    for (int pair_22 = 2; pair_22 < 16; pair_22 += 2) {
                        float2 _f2_70 = make_float2((_tmem_load_2 + 96)[pair_22], (_tmem_load_2 + 96)[pair_22 + 1]);
                        partial_112_1 = add_f32x2(partial_112_1, _f2_70);
                    }
                    float2 frag_sum2_113_1 = partial_112_1;
                    float2 _f2_71 = make_float2(((group_max6 > -WAN_HYBRID_INF) ? _exp2_24 : 0.0f), ((group_max6 > -WAN_HYBRID_INF) ? _exp2_24 : 0.0f));
                    float2 raw_scale2_114_1 = _f2_71;
                    float2 raw_sum2_115_1 = mul_f32x2(frag_sum2_113_1, raw_scale2_114_1);
                    uint32_t _fp4_22[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_22[0]) : "f"(_tmem_load_2[96]), "f"(_tmem_load_2[97]), "f"(_tmem_load_2[98]), "f"(_tmem_load_2[99]), "f"(_tmem_load_2[100]), "f"(_tmem_load_2[101]), "f"(_tmem_load_2[102]), "f"(_tmem_load_2[103]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_22[1]) : "f"(_tmem_load_2[104]), "f"(_tmem_load_2[105]), "f"(_tmem_load_2[106]), "f"(_tmem_load_2[107]), "f"(_tmem_load_2[108]), "f"(_tmem_load_2[109]), "f"(_tmem_load_2[110]), "f"(_tmem_load_2[111]));
                    block_sum2_70_1 = add_f32x2(block_sum2_70_1, raw_sum2_115_1);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72_1 + 72 + (unsigned int)(warp % 4 * 32 << 16) + 4), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_22[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_22[1]))
                        : "memory");
                    float _exp2_25 = approx_exp2(((group_max7 > -WAN_HYBRID_INF) ? group_max7 * softmax_scale_log2 : 0.0f) - new_max_scaled_69_1 - 2.584962500721156f);
                    sf_values_71_1[3] = ((group_max7 > -WAN_HYBRID_INF) ? _exp2_25 : 0.0f);
                    const float2 _fma_b2_62 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_63 = {2.584962500721156f - ((group_max7 > -WAN_HYBRID_INF) ? group_max7 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((group_max7 > -WAN_HYBRID_INF) ? group_max7 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_2 + 112))[_lf], _fma_b2_62, _fma_c2_63);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        _tmem_load_2[_le + 112] = approx_exp2(_tmem_load_2[_le + 112]);
                    }
                    float2 _f2_72 = make_float2(_tmem_load_2[112], _tmem_load_2[113]);
                    float2 partial_116_1 = _f2_72;
                    #pragma unroll
                    for (int pair_23 = 2; pair_23 < 16; pair_23 += 2) {
                        float2 _f2_73 = make_float2((_tmem_load_2 + 112)[pair_23], (_tmem_load_2 + 112)[pair_23 + 1]);
                        partial_116_1 = add_f32x2(partial_116_1, _f2_73);
                    }
                    float2 frag_sum2_117_1 = partial_116_1;
                    float2 _f2_74 = make_float2(((group_max7 > -WAN_HYBRID_INF) ? _exp2_25 : 0.0f), ((group_max7 > -WAN_HYBRID_INF) ? _exp2_25 : 0.0f));
                    float2 raw_scale2_118_1 = _f2_74;
                    float2 raw_sum2_119_1 = mul_f32x2(frag_sum2_117_1, raw_scale2_118_1);
                    uint32_t _fp4_23[2];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_23[0]) : "f"(_tmem_load_2[112]), "f"(_tmem_load_2[113]), "f"(_tmem_load_2[114]), "f"(_tmem_load_2[115]), "f"(_tmem_load_2[116]), "f"(_tmem_load_2[117]), "f"(_tmem_load_2[118]), "f"(_tmem_load_2[119]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_23[1]) : "f"(_tmem_load_2[120]), "f"(_tmem_load_2[121]), "f"(_tmem_load_2[122]), "f"(_tmem_load_2[123]), "f"(_tmem_load_2[124]), "f"(_tmem_load_2[125]), "f"(_tmem_load_2[126]), "f"(_tmem_load_2[127]));
                    block_sum2_70_1 = add_f32x2(block_sum2_70_1, raw_sum2_119_1);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72_1 + 72 + (unsigned int)(warp % 4 * 32 << 16) + 6), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_23[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_23[1]))
                        : "memory");
                    uint32_t _fp8_5[1];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sf_values_71_1[0]), "f"(sf_values_71_1[1]),
                                               "f"(sf_values_71_1[2]), "f"(sf_values_71_1[3]));
                        _fp8_5[0] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)p_stage_off_72_1 + 84 + (unsigned int)(warp % 4)), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_5[0]))
                        : "memory");
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_2_addr + (stage) * 8);
                    row_sum_val = row_sum_val * acc_scale_1 + (block_sum2_70_1.x + block_sum2_70_1.y);
                }
                mbarrier_wait(o_full_addr + (stage) * 8, _phase_o_full);
                _phase_o_full ^= 1;
                sScale[warp % 4 * 32 + lane + scale_off] = row_sum_val;
                mbarrier_arrive(corr_sig_addr + (stage) * 8);
            }
        }
    // ---- Role: correction ----
    } else if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
        { // correction_main
            unsigned int total_tiles_1 = NUM_M_BLOCKS * total_bh;
            unsigned int corr_epi_producer_phase = 1;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = bid; tile_idx_1 < total_tiles_1; tile_idx_1 += num_bids) {
                unsigned int m_block_1;
                unsigned int bh_1;
                {
                    m_block_1 = tile_idx_1 % (unsigned int)NUM_M_BLOCKS;
                    bh_1 = tile_idx_1 / (unsigned int)NUM_M_BLOCKS;
                }
                unsigned int num_n_blocks_1 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                int off_q0 = bh_1 * (unsigned int)seqlen_q + m_block_1 * 2 * (unsigned int)BLOCK_M;
                int off_q1 = off_q0 + BLOCK_M;
                mbarrier_arrive(p_full_addr);
                mbarrier_arrive(p_full_addr + 8);
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                #pragma unroll 1
                for (unsigned int n_iter_2 = 1; n_iter_2 < num_n_blocks_1; n_iter_2++) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    float scale = sScale[warp % 4 * 32 + lane];
                    int _vote_0 = __any_sync(0xFFFFFFFF, scale < 1.0f);
                    if (_vote_0 != 0) {
                        #pragma unroll
                        for (int cr_col = 0; cr_col < HEAD_DIM / 16; cr_col++) {
                            int cr_addr = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(cr_col * 16);
                            float _tmem_load_3[16];
                            tmem_ld_x16(&_tmem_load_3[0], cr_addr);
                            const float2 _scale2_0 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_0);
                            tmem_st_x16_f32(cr_addr, _tmem_load_3);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr);
                    mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                    _phase_corr_sig_1 ^= 1;
                    float scale1 = sScale[warp % 4 * 32 + lane + BLOCK_M];
                    int _vote_1 = __any_sync(0xFFFFFFFF, scale1 < 1.0f);
                    if (_vote_1 != 0) {
                        #pragma unroll
                        for (int cr_col_1 = 0; cr_col_1 < HEAD_DIM / 16; cr_col_1++) {
                            int cr_addr_1 = taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(cr_col_1 * 16);
                            float _tmem_load_4[16];
                            tmem_ld_x16(&_tmem_load_4[0], cr_addr_1);
                            const float2 _scale2_1 = {scale1, scale1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_1);
                            tmem_st_x16_f32(cr_addr_1, _tmem_load_4);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr + 8);
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                _phase_o_full_1 ^= 1;
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                #pragma unroll
                for (int stage_1 = 0; stage_1 < 2; stage_1++) {
                    int tmem_o_off = ((stage_1 == 0) ? TMEM_OUTPUT_0_OFFSET : TMEM_OUTPUT_1_OFFSET);
                    int s_off = stage_1 * BLOCK_M;
                    int off_q = ((stage_1 == 0) ? off_q0 : off_q1);
                    mbarrier_wait(epi_empty_addr + (stage_1) * 8, corr_epi_producer_phase);
                    float final_sum = sScale[warp % 4 * 32 + lane + s_off];
                    float final_scale;
                    if (final_sum != 0.0f && final_sum == final_sum) {
                        float _rcp_0 = approx_rcp(final_sum);
                        final_scale = _rcp_0;
                    } else {
                        final_scale = 0.0f;
                    }
                    int ce_warp_row_base = warp % 4 * 32;
                    int ce_matrix = lane / 8;
                    int ce_matrix_row = lane & 7;
                    int ce_stage_base = smem_o_addr + (unsigned int)(stage_1 * 32768);
                    #pragma unroll
                    for (int ce_row_half = 0; ce_row_half < 2; ce_row_half++) {
                        int ce_tmem_row = (warp % 4 * 32 << 16) + (ce_row_half * 16 << 16);
                        int ce_query_base = ce_warp_row_base + ce_row_half * 16;
                        #pragma unroll
                        for (int ce_col_group = 0; ce_col_group < HEAD_DIM / 32; ce_col_group++) {
                            int ce_col_base = ce_col_group * 32;
                            int ce_addr = taddr + (unsigned int)tmem_o_off + (unsigned int)ce_tmem_row + (unsigned int)ce_col_base;
                            float _tmem_load_5[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                                : "r"(ce_addr)
                                : "memory");
                            int ce_scale_lane = ce_row_half * 16 + lane / 4;
                            float _shfl_0 = __shfl_sync(0xFFFFFFFF, final_scale, ce_scale_lane);
                            float ce_scale_top = _shfl_0;
                            float _shfl_1 = __shfl_sync(0xFFFFFFFF, final_scale, ce_scale_lane + 8);
                            float ce_scale_bottom = _shfl_1;
                            #pragma unroll
                            for (int ce_repeat = 0; ce_repeat < 4; ce_repeat++) {
                                const int ce_repeat_base = ce_repeat * 4;
                                const float2 _scale2_2 = {ce_scale_top, ce_scale_top};
                                #pragma unroll
                                for (int _ls = 0; _ls < 1; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_5 + ce_repeat_base))[_ls], _scale2_2);
                                const float2 _scale2_3 = {ce_scale_bottom, ce_scale_bottom};
                                #pragma unroll
                                for (int _ls = 0; _ls < 1; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_5 + ce_repeat_base + 2))[_ls], _scale2_3);
                            }
                            unsigned int ce_packed[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                                ce_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                            int ce_matrix_col = ce_col_base + ce_matrix * 8;
                            int ce_atom_row = ce_matrix_col / 64 * BLOCK_M + ce_query_base + ce_matrix_row;
                            int ce_atom_col_bytes = ce_matrix_col % 64 * 2;
                            uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)(ce_stage_base + (ce_atom_row * 128 + ce_atom_col_bytes ^ (ce_atom_row * 128 + ce_atom_col_bytes >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&ce_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&ce_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&ce_packed[4])), "r"(*reinterpret_cast<const uint32_t*>(&ce_packed[6]))
                                : "memory");
                            uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)(ce_stage_base + ((ce_atom_row + 8) * 128 + ce_atom_col_bytes ^ ((ce_atom_row + 8) * 128 + ce_atom_col_bytes >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&ce_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&ce_packed[3])), "r"(*reinterpret_cast<const uint32_t*>(&ce_packed[5])), "r"(*reinterpret_cast<const uint32_t*>(&ce_packed[7]))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    mbarrier_arrive(epi_full_addr + (stage_1) * 8);
                }
                corr_epi_producer_phase ^= 1;
            }
        }
    // ---- Role: mma ----
    } else if (warp == 12) {
        { // mma_main
            unsigned int total_tiles_2 = NUM_M_BLOCKS * total_bh;
            unsigned int mma_kv_stage = 0;
            unsigned int mma_kv_phase = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_2_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_p_full_2_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = bid; tile_idx_2 < total_tiles_2; tile_idx_2 += num_bids) {
                unsigned int m_block_2;
                unsigned int bh_2;
                {
                    m_block_2 = tile_idx_2 % (unsigned int)NUM_M_BLOCKS;
                    bh_2 = tile_idx_2 / (unsigned int)NUM_M_BLOCKS;
                }
                unsigned int num_n_blocks_2 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase);
                int _mma_a_lo_0 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
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
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores_0), "r"(0));
                elect_commit(s_full_addr);
                mbarrier_wait(q_full_addr + 8, _phase_q_full_1);
                _phase_q_full_1 ^= 1;
                int _mma_a_lo_1 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                int _mma_b_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
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
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_scores_1), "r"(0));
                elect_commit(s_full_addr + 8);
                elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                mma_kv_stage += 1;
                if (mma_kv_stage == 3) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                unsigned int first_pv = 1;
                #pragma unroll 1
                for (unsigned int n_iter_3 = 0; n_iter_3 < num_n_blocks_2 - 1; n_iter_3++) {
                    unsigned int v_stage = mma_kv_stage;
                    unsigned int v_phase = mma_kv_phase;
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 3) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
                    int first_pv_flag = first_pv;
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv0_lo, make_sf_cp_desc_sbo128(smem_sfvt_lo_addr + v_stage * 32768));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv0_res_lo, make_sf_cp_desc_sbo128(smem_sfvt_residual_lo_addr + v_stage * 32768));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv0_hi, make_sf_cp_desc_sbo128(smem_sfvt_hi_addr + v_stage * 32768));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv0_res_hi, make_sf_cp_desc_sbo128(smem_sfvt_residual_hi_addr + v_stage * 32768));
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_2 = make_warp_uniform((((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 2048);
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_2) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_0, tmem_scores_0 + 64 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv0_lo + 0, ((first_pv_flag) ? 0 : 1));
                        }
                    }
                    int _mma_b_lo_3 = make_warp_uniform((((smem_vt_residual_addr) >> 4) & 0x3FFF) + (v_stage) * 2048);
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_3) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_0, tmem_scores_0 + 64 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv0_res_lo + 0, 1);
                        }
                    }
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                    _phase_p_full_2_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_4 = make_warp_uniform((((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 2048);
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_4) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_0, tmem_scores_0 + 72 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv0_hi + 0, 1);
                        }
                    }
                    int _mma_b_lo_5 = make_warp_uniform((((smem_vt_residual_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 2048);
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_5) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_0, tmem_scores_0 + 72 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv0_res_hi + 0, 1);
                        }
                    }
                    unsigned int k_stage = mma_kv_stage;
                    unsigned int k_phase = mma_kv_phase;
                    mma_kv_stage += 1;
                    if (mma_kv_stage == 3) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                    int _mma_a_lo_6 = make_warp_uniform(((smem_q0_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_6 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
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
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_scores_0), "r"(0));
                    elect_commit(s_full_addr);
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv1_lo, make_sf_cp_desc_sbo128(smem_sfvt_lo_addr + v_stage * 32768));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv1_res_lo, make_sf_cp_desc_sbo128(smem_sfvt_residual_lo_addr + v_stage * 32768));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv1_hi, make_sf_cp_desc_sbo128(smem_sfvt_hi_addr + v_stage * 32768));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv1_res_hi, make_sf_cp_desc_sbo128(smem_sfvt_residual_hi_addr + v_stage * 32768));
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_7 = make_warp_uniform((((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 2048);
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_7) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_1, tmem_scores_1 + 64 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv1_lo + 0, ((first_pv_flag) ? 0 : 1));
                        }
                    }
                    int _mma_b_lo_8 = make_warp_uniform((((smem_vt_residual_addr) >> 4) & 0x3FFF) + (v_stage) * 2048);
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_8) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_1, tmem_scores_1 + 64 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv1_res_lo + 0, 1);
                        }
                    }
                    mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                    _phase_p_full_2_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_9 = make_warp_uniform((((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 2048);
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_9) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_1, tmem_scores_1 + 72 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv1_hi + 0, 1);
                        }
                    }
                    int _mma_b_lo_10 = make_warp_uniform((((smem_vt_residual_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 2048);
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_10) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_1, tmem_scores_1 + 72 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv1_res_hi + 0, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (v_stage) * 8);
                    int _mma_a_lo_11 = make_warp_uniform(((smem_q1_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_11 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
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
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"(tmem_scores_1), "r"(0));
                    elect_commit(s_full_addr + 8);
                    elect_commit(kv_empty_addr + (k_stage) * 8);
                    first_pv = 0;
                }
                elect_commit(q_empty_addr);
                elect_commit(q_empty_addr + 8);
                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase);
                int first_pv_flag_1 = first_pv;
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv0_lo, make_sf_cp_desc_sbo128(smem_sfvt_lo_addr + mma_kv_stage * 32768));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv0_res_lo, make_sf_cp_desc_sbo128(smem_sfvt_residual_lo_addr + mma_kv_stage * 32768));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv0_hi, make_sf_cp_desc_sbo128(smem_sfvt_hi_addr + mma_kv_stage * 32768));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv0_res_hi, make_sf_cp_desc_sbo128(smem_sfvt_residual_hi_addr + mma_kv_stage * 32768));
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_12 = make_warp_uniform((((smem_vt_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                if (elect_sync()) {
                    {
                        uint64_t b_desc = ((uint64_t)_mma_b_lo_12) | ((uint64_t)0x80004020 << 32);
                        tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_0, tmem_scores_0 + 64 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv0_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                    }
                }
                int _mma_b_lo_13 = make_warp_uniform((((smem_vt_residual_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                if (elect_sync()) {
                    {
                        uint64_t b_desc = ((uint64_t)_mma_b_lo_13) | ((uint64_t)0x80004020 << 32);
                        tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_0, tmem_scores_0 + 64 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv0_res_lo + 0, 1);
                    }
                }
                mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                _phase_p_full_2_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_14 = make_warp_uniform((((smem_vt_addr + 32) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                if (elect_sync()) {
                    {
                        uint64_t b_desc = ((uint64_t)_mma_b_lo_14) | ((uint64_t)0x80004020 << 32);
                        tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_0, tmem_scores_0 + 72 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv0_hi + 0, 1);
                    }
                }
                int _mma_b_lo_15 = make_warp_uniform((((smem_vt_residual_addr + 32) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                if (elect_sync()) {
                    {
                        uint64_t b_desc = ((uint64_t)_mma_b_lo_15) | ((uint64_t)0x80004020 << 32);
                        tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_0, tmem_scores_0 + 72 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv0_res_hi + 0, 1);
                    }
                }
                mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                _phase_p_full_1 ^= 1;
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv1_lo, make_sf_cp_desc_sbo128(smem_sfvt_lo_addr + mma_kv_stage * 32768));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv1_res_lo, make_sf_cp_desc_sbo128(smem_sfvt_residual_lo_addr + mma_kv_stage * 32768));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv1_hi, make_sf_cp_desc_sbo128(smem_sfvt_hi_addr + mma_kv_stage * 32768));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb_pv1_res_hi, make_sf_cp_desc_sbo128(smem_sfvt_residual_hi_addr + mma_kv_stage * 32768));
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_16 = make_warp_uniform((((smem_vt_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                if (elect_sync()) {
                    {
                        uint64_t b_desc = ((uint64_t)_mma_b_lo_16) | ((uint64_t)0x80004020 << 32);
                        tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_1, tmem_scores_1 + 64 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv1_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                    }
                }
                int _mma_b_lo_17 = make_warp_uniform((((smem_vt_residual_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                if (elect_sync()) {
                    {
                        uint64_t b_desc = ((uint64_t)_mma_b_lo_17) | ((uint64_t)0x80004020 << 32);
                        tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_1, tmem_scores_1 + 64 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv1_res_lo + 0, 1);
                    }
                }
                mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                _phase_p_full_2_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_18 = make_warp_uniform((((smem_vt_addr + 32) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                if (elect_sync()) {
                    {
                        uint64_t b_desc = ((uint64_t)_mma_b_lo_18) | ((uint64_t)0x80004020 << 32);
                        tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_1, tmem_scores_1 + 72 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv1_hi + 0, 1);
                    }
                }
                int _mma_b_lo_19 = make_warp_uniform((((smem_vt_residual_addr + 32) >> 4) & 0x3FFF) + (mma_kv_stage) * 2048);
                if (elect_sync()) {
                    {
                        uint64_t b_desc = ((uint64_t)_mma_b_lo_19) | ((uint64_t)0x80004020 << 32);
                        tcgen05_mma_mxf4nvf4_bs_ts(tmem_output_1, tmem_scores_1 + 72 + 0, b_desc + 0, 0x8200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv1_res_hi + 0, 1);
                    }
                }
                elect_commit(kv_empty_addr + (mma_kv_stage) * 8);
                mma_kv_stage += 1;
                if (mma_kv_stage == 3) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                elect_commit2(o_full_addr, o_full_addr + 8);
            }
        }
    // ---- Role: epilogue ----
    } else if (warp == 13) {
        { // epilogue_main
            unsigned int total_tiles_3 = NUM_M_BLOCKS * total_bh;
            unsigned int epi_consumer_phase = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_3 = bid; tile_idx_3 < total_tiles_3; tile_idx_3 += num_bids) {
                unsigned int m_block_3;
                unsigned int bh_3;
                {
                    m_block_3 = tile_idx_3 % (unsigned int)NUM_M_BLOCKS;
                    bh_3 = tile_idx_3 / (unsigned int)NUM_M_BLOCKS;
                }
                unsigned int num_n_blocks_3 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                int head = bh_3 % (unsigned int)heads;
                int batch_idx = bh_3 / (unsigned int)heads;
                #pragma unroll
                for (int stage_2 = 0; stage_2 < 2; stage_2++) {
                    mbarrier_wait(epi_full_addr + (stage_2) * 8, epi_consumer_phase);
                    int q_row = m_block_3 * 2 * (unsigned int)BLOCK_M + (unsigned int)(stage_2 * BLOCK_M);
                    if (elect_sync()) {
                        tma_store_5d(O, 0, q_row, head, batch_idx, 0, smem_o_addr + (unsigned int)(stage_2 * 32768));
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                asm volatile("cp.async.bulk.wait_group.read 1;");
                mbarrier_arrive(epi_empty_addr);
                asm volatile("cp.async.bulk.wait_group.read 0;");
                mbarrier_arrive(epi_empty_addr + 8);
                epi_consumer_phase ^= 1;
            }
        }
    // ---- Role: load ----
    } else if (warp == 14) {
        { // load_main
            unsigned int total_tiles_4 = NUM_M_BLOCKS * total_bh;
            unsigned int num_n_blocks_all = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
            unsigned int v_level_row_stride = total_bh * HEAD_DIM;
            unsigned int v_scale_level_stride = total_bh * physical_num_blocks * 16;
            unsigned int load_kv_stage = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_q_empty_1 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_4 = bid; tile_idx_4 < total_tiles_4; tile_idx_4 += num_bids) {
                unsigned int m_block_4;
                unsigned int bh_4;
                {
                    m_block_4 = tile_idx_4 % (unsigned int)NUM_M_BLOCKS;
                    bh_4 = tile_idx_4 / (unsigned int)NUM_M_BLOCKS;
                }
                unsigned int num_n_blocks_4 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                int head_1 = bh_4 % (unsigned int)heads;
                int batch_idx_1 = bh_4 / (unsigned int)heads;
                int off_q0_1 = m_block_4 * 2 * (unsigned int)BLOCK_M;
                int off_q1_1 = off_q0_1 + BLOCK_M;
                int off_kv = 0;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 32768);
                    tma_5d_gmem2smem(smem_q0_addr, Q, 0, off_q0_1, head_1, batch_idx_1, 0, q_full_addr);
                }
                int first_kv_off = (unsigned int)off_kv + (num_n_blocks_4 - 1) * (unsigned int)BLOCK_N;
                mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 32768);
                    tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 32768, K, 0, first_kv_off, head_1, batch_idx_1, 0, kv_full_addr + (load_kv_stage) * 8);
                }
                load_kv_stage += 1;
                if (load_kv_stage == 3) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                mbarrier_wait(q_empty_addr + 8, _phase_q_empty_1);
                _phase_q_empty_1 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + 8, 32768);
                    tma_5d_gmem2smem(smem_q1_addr, Q, 0, off_q1_1, head_1, batch_idx_1, 0, q_full_addr + 8);
                }
                mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    int first_kv_sf_tile = bh_4 * (unsigned int)physical_num_blocks + num_n_blocks_4 - 1;
                    int first_vt_row = bh_4 * (unsigned int)HEAD_DIM;
                    int first_vt_col = (num_n_blocks_4 - 1) * (unsigned int)(BLOCK_N / 2);
                    tma_2d_gmem2smem(smem_vt_addr + load_kv_stage * 32768, Vt, first_vt_col, first_vt_row, kv_full_addr + (load_kv_stage) * 8);
                    tma_2d_gmem2smem(smem_vt_residual_addr + load_kv_stage * 32768, Vt, first_vt_col, v_level_row_stride + (unsigned int)first_vt_row, kv_full_addr + (load_kv_stage) * 8);
                    tma_2d_gmem2smem(smem_sfvt_lo_addr + load_kv_stage * 32768, SFVtLo, 0, first_kv_sf_tile * 16, kv_full_addr + (load_kv_stage) * 8);
                    tma_2d_gmem2smem(smem_sfvt_residual_lo_addr + load_kv_stage * 32768, SFVtLo, 0, v_scale_level_stride + (unsigned int)(first_kv_sf_tile * 16), kv_full_addr + (load_kv_stage) * 8);
                    tma_2d_gmem2smem(smem_sfvt_hi_addr + load_kv_stage * 32768, SFVtHi, 0, first_kv_sf_tile * 16, kv_full_addr + (load_kv_stage) * 8);
                    tma_2d_gmem2smem(smem_sfvt_residual_hi_addr + load_kv_stage * 32768, SFVtHi, 0, v_scale_level_stride + (unsigned int)(first_kv_sf_tile * 16), kv_full_addr + (load_kv_stage) * 8);
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 18432);
                }
                load_kv_stage += 1;
                if (load_kv_stage == 3) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                #pragma unroll 1
                for (unsigned int ni = 1; ni < num_n_blocks_4; ni++) {
                    unsigned int n = num_n_blocks_4 - 1 - ni;
                    int kv_off = (unsigned int)off_kv + n * (unsigned int)BLOCK_N;
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 32768);
                        tma_5d_gmem2smem(smem_kv_addr + load_kv_stage * 32768, K, 0, kv_off, head_1, batch_idx_1, 0, kv_full_addr + (load_kv_stage) * 8);
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 3) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        int kv_sf_tile = bh_4 * (unsigned int)physical_num_blocks + n;
                        int vt_row = bh_4 * (unsigned int)HEAD_DIM;
                        int vt_col = n * (unsigned int)(BLOCK_N / 2);
                        tma_2d_gmem2smem(smem_vt_addr + load_kv_stage * 32768, Vt, vt_col, vt_row, kv_full_addr + (load_kv_stage) * 8);
                        tma_2d_gmem2smem(smem_vt_residual_addr + load_kv_stage * 32768, Vt, vt_col, v_level_row_stride + (unsigned int)vt_row, kv_full_addr + (load_kv_stage) * 8);
                        tma_2d_gmem2smem(smem_sfvt_lo_addr + load_kv_stage * 32768, SFVtLo, 0, kv_sf_tile * 16, kv_full_addr + (load_kv_stage) * 8);
                        tma_2d_gmem2smem(smem_sfvt_residual_lo_addr + load_kv_stage * 32768, SFVtLo, 0, v_scale_level_stride + (unsigned int)(kv_sf_tile * 16), kv_full_addr + (load_kv_stage) * 8);
                        tma_2d_gmem2smem(smem_sfvt_hi_addr + load_kv_stage * 32768, SFVtHi, 0, kv_sf_tile * 16, kv_full_addr + (load_kv_stage) * 8);
                        tma_2d_gmem2smem(smem_sfvt_residual_hi_addr + load_kv_stage * 32768, SFVtHi, 0, v_scale_level_stride + (unsigned int)(kv_sf_tile * 16), kv_full_addr + (load_kv_stage) * 8);
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_kv_stage) * 8, 18432);
                    }
                    load_kv_stage += 1;
                    if (load_kv_stage == 3) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                }
            }
        }
    // ---- Role: empty ----
    } else if (warp == 15) {
        // idle — no tasks assigned
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    {
        const int warp = make_warp_uniform(tid / 32);
        if (warp == 12) {
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
}

} // extern "C"
