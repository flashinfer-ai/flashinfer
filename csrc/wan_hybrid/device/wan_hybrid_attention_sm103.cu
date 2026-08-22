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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define WAN_HYBRID_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_SCORES_OFFSET 0
#define TMEM_OUTPUT_0_OFFSET 256
#define TMEM_OUTPUT_1_OFFSET 384
#define TMEM_TMEM_SFA_QK0_OFFSET 424
#define TMEM_TMEM_SFB_QK0_OFFSET 432
#define TMEM_TMEM_SFA_QK1_OFFSET 440
#define TMEM_TMEM_SFA_PV0_LO_OFFSET 16
#define TMEM_TMEM_SFA_PV0_HI_OFFSET 20
#define TMEM_TMEM_SFA_PV1_LO_OFFSET 144
#define TMEM_TMEM_SFA_PV1_HI_OFFSET 148
#define TMEM_TMEM_SFA_PV0_RES_LO_OFFSET 28
#define TMEM_TMEM_SFA_PV0_RES_HI_OFFSET 36
#define TMEM_TMEM_SFA_PV1_RES_LO_OFFSET 408
#define TMEM_TMEM_SFA_PV1_RES_HI_OFFSET 412
#define TMEM_TMEM_SFB_PV_LO_OFFSET 24
#define TMEM_TMEM_SFB_PV_HI_OFFSET 32
#define NUM_K_PIPE_STAGES 2
#define NUM_V_PIPE_STAGES 4
#define SMEM_ROW_STATE_OFF 1024
#define SMEM_ROW_STATE_STAGE_BYTES 2048
#define SMEM_ROW_STATE_STRIDE 2048
#define SMEM_SMEM_Q_OFF 3072
#define SMEM_SMEM_Q_STAGE_BYTES 32768
#define SMEM_SMEM_Q_STRIDE 32768
#define SMEM_SMEM_SFQ_OFF 1024
#define SMEM_SMEM_SFQ_STAGE_BYTES 1024
#define SMEM_SMEM_SFQ_STRIDE 1024
#define SMEM_SMEM_K_OFF 68608
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_VT_OFF 101376
#define SMEM_SMEM_VT_STAGE_BYTES 4096
#define SMEM_SMEM_VT_STRIDE 4096
#define SMEM_SMEM_VT_RESIDUAL_OFF 117760
#define SMEM_SMEM_VT_RESIDUAL_STAGE_BYTES 4096
#define SMEM_SMEM_VT_RESIDUAL_STRIDE 4096
#define SMEM_SMEM_SFK_OFF 1024
#define SMEM_SMEM_SFK_STAGE_BYTES 1024
#define SMEM_SMEM_SFK_STRIDE 1024
#define SMEM_SMEM_SFVT_LO_OFF 134144
#define SMEM_SMEM_SFVT_LO_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_LO_STRIDE 1024
#define SMEM_SMEM_SFVT_HI_OFF 134656
#define SMEM_SMEM_SFVT_HI_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_HI_STRIDE 1024
#define SMEM_SMEM_SFVT_RESIDUAL_LO_OFF 138240
#define SMEM_SMEM_SFVT_RESIDUAL_LO_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_RESIDUAL_LO_STRIDE 1024
#define SMEM_SMEM_SFVT_RESIDUAL_HI_OFF 138752
#define SMEM_SMEM_SFVT_RESIDUAL_HI_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_RESIDUAL_HI_STRIDE 1024
#define SMEM_SMEM_P_OFF 142336
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_SMEM_P_RESIDUAL_OFF 142336
#define SMEM_SMEM_P_RESIDUAL_STAGE_BYTES 8192
#define SMEM_SMEM_P_RESIDUAL_STRIDE 8192
#define SMEM_SMEM_SFP_CP_LO_OFF 1024
#define SMEM_SMEM_SFP_CP_LO_STAGE_BYTES 512
#define SMEM_SMEM_SFP_CP_LO_STRIDE 1024
#define SMEM_SMEM_SFP_CP_HI_OFF 1536
#define SMEM_SMEM_SFP_CP_HI_STAGE_BYTES 512
#define SMEM_SMEM_SFP_CP_HI_STRIDE 1024
#define SMEM_SMEM_O00_OFF 158720
#define SMEM_SMEM_O00_STAGE_BYTES 32768
#define SMEM_SMEM_O00_STRIDE 32768
#define SMEM_SMEM_O01_OFF 175104
#define SMEM_SMEM_O01_STAGE_BYTES 16384
#define SMEM_SMEM_O01_STRIDE 16384
#define SMEM_SMEM_O10_OFF 191488
#define SMEM_SMEM_O10_STAGE_BYTES 32768
#define SMEM_SMEM_O10_STRIDE 32768
#define SMEM_SMEM_O11_OFF 207872
#define SMEM_SMEM_O11_STAGE_BYTES 16384
#define SMEM_SMEM_O11_STRIDE 16384
#define SMEM_SMEM_QK_CORRECTION_OFF 1024
#define SMEM_SMEM_QK_CORRECTION_STAGE_BYTES 1024
#define SMEM_SMEM_QK_CORRECTION_STRIDE 1024
#define SMEM_TOTAL 224256
#define IS_CAUSAL 0
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128
#define HAS_QK_CORRECTION 0
#define num_m_blocks (((seqlen_q + (4 * BLOCK_M)) - 1) / (4 * BLOCK_M))

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


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_ts_cta2(
    int taddr, int a_taddr, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], [%1], %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "r"(a_taddr), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ void elect_commit_cg2_local(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


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


__device__ __forceinline__ void tmem_st_x8_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]),
           "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7]));
}


__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}


__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
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


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
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


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo128(int addr) {
    const int SBO = 128;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo256(int addr) {
    const int SBO = 256;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_5d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
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


__device__ __forceinline__ void tcgen05_commit_cg2_local(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}

extern "C" {

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_wan_hybrid_attention(WanHybridTensorMap const* Q, WanHybridTensorMap const* K, WanHybridTensorMap const* Vt, WanHybridTensorMap const* SFQ, WanHybridTensorMap const* SFK, WanHybridTensorMap const* SFVtLo, WanHybridTensorMap const* SFVtHi, float* __restrict__ QKCorrection, WanHybridTensorMap const* O, int seqlen_q, int seqlen_kv, int q_stride, int kv_stride, float softmax_scale_log2, int heads, int total_bh)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Vt)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFQ)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFK)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFVtLo)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFVtHi)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(O)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* row_state = reinterpret_cast<float*>(smem_raw + 1024);
    const int row_state_addr = smem + 1024;
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_q_addr = smem + 3072;
    uint8_t* smem_sfq = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_sfq_addr = smem + 1024;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int smem_k_addr = smem + 68608;
    uint8_t* smem_vt = reinterpret_cast<uint8_t*>(smem_raw + 101376);
    const int smem_vt_addr = smem + 101376;
    uint8_t* smem_vt_residual = reinterpret_cast<uint8_t*>(smem_raw + 117760);
    const int smem_vt_residual_addr = smem + 117760;
    uint8_t* smem_sfk = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_sfk_addr = smem + 1024;
    uint8_t* smem_sfvt_lo = reinterpret_cast<uint8_t*>(smem_raw + 134144);
    const int smem_sfvt_lo_addr = smem + 134144;
    uint8_t* smem_sfvt_hi = reinterpret_cast<uint8_t*>(smem_raw + 134656);
    const int smem_sfvt_hi_addr = smem + 134656;
    uint8_t* smem_sfvt_residual_lo = reinterpret_cast<uint8_t*>(smem_raw + 138240);
    const int smem_sfvt_residual_lo_addr = smem + 138240;
    uint8_t* smem_sfvt_residual_hi = reinterpret_cast<uint8_t*>(smem_raw + 138752);
    const int smem_sfvt_residual_hi_addr = smem + 138752;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 142336);
    const int smem_p_addr = smem + 142336;
    uint8_t* smem_p_residual = reinterpret_cast<uint8_t*>(smem_raw + 142336);
    const int smem_p_residual_addr = smem + 142336;
    uint8_t* smem_sfp_cp_lo = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_sfp_cp_lo_addr = smem + 1024;
    uint8_t* smem_sfp_cp_hi = reinterpret_cast<uint8_t*>(smem_raw + 1536);
    const int smem_sfp_cp_hi_addr = smem + 1536;
    __nv_bfloat16* smem_o00 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 158720);
    const int smem_o00_addr = smem + 158720;
    __nv_bfloat16* smem_o01 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 175104);
    const int smem_o01_addr = smem + 175104;
    __nv_bfloat16* smem_o10 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 191488);
    const int smem_o10_addr = smem + 191488;
    __nv_bfloat16* smem_o11 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 207872);
    const int smem_o11_addr = smem + 207872;
    float* smem_qk_correction = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_qk_correction_addr = smem + 1024;

    // Mbarrier init (17 groups, 33 barriers)
    // Mbarriers at smem_raw[0..264)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 2 barriers, init_count=2
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            // k_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 4 barriers, init_count=2
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            // v_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 136, 256);
            mbarrier_init(smem + 144, 256);
            // p_full_2: 2 barriers, init_count=256
            mbarrier_init(smem + 152, 256);
            mbarrier_init(smem + 160, 256);
            // pv_reads_done: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            // corr_sig: 2 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            mbarrier_init(smem + 200, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // o_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 224, 256);
            mbarrier_init(smem + 232, 256);
            // tile_done: 1 barriers, init_count=256
            mbarrier_init(smem + 240, 256);
            // peer_tile_done: 1 barriers, init_count=2
            mbarrier_init(smem + 248, 2);
            // tile_start: 1 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 264);
    if (warp == 0) {
        int _tmem_hold = smem + 264;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define k_full_addr (mbar_base + 24)
    #define k_empty_addr (mbar_base + 40)
    #define v_full_addr (mbar_base + 56)
    #define v_empty_addr (mbar_base + 88)
    #define s_full_addr (mbar_base + 120)
    #define p_full_addr (mbar_base + 136)
    #define p_full_2_addr (mbar_base + 152)
    #define pv_reads_done_addr (mbar_base + 168)
    #define corr_sig_addr (mbar_base + 176)
    #define corr_done_addr (mbar_base + 192)
    #define o_full_addr (mbar_base + 208)
    #define o_empty_addr (mbar_base + 224)
    #define tile_done_addr (mbar_base + 240)
    #define peer_tile_done_addr (mbar_base + 248)
    #define tile_start_addr (mbar_base + 256)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_output_0 = taddr + 256;
    const int tmem_output_1 = taddr + 384;
    const int tmem_tmem_sfa_qk0 = taddr + 424;
    const int tmem_tmem_sfb_qk0 = taddr + 432;
    const int tmem_tmem_sfa_qk1 = taddr + 440;
    const int tmem_tmem_sfa_pv0_lo = taddr + 16;
    const int tmem_tmem_sfa_pv0_hi = taddr + 20;
    const int tmem_tmem_sfa_pv1_lo = taddr + 144;
    const int tmem_tmem_sfa_pv1_hi = taddr + 148;
    const int tmem_tmem_sfa_pv0_res_lo = taddr + 28;
    const int tmem_tmem_sfa_pv0_res_hi = taddr + 36;
    const int tmem_tmem_sfa_pv1_res_lo = taddr + 408;
    const int tmem_tmem_sfa_pv1_res_hi = taddr + 412;
    const int tmem_tmem_sfb_pv_lo = taddr + 24;
    const int tmem_tmem_sfb_pv_hi = taddr + 32;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // softmax_main
            unsigned int total_tiles = num_m_blocks * total_bh;
            unsigned int stage = make_warp_uniform(warp / 4);
            int scale_off = make_warp_uniform(stage * (unsigned int)BLOCK_M);
            unsigned int _phase_tile_start_0 = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_corr_done = 0;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_o_full = 0;
            unsigned int _phase_tile_done_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < total_tiles; tile_idx += num_clusters) {
                mbarrier_wait(tile_start_addr, _phase_tile_start_0);
                _phase_tile_start_0 ^= 1;
                unsigned int cluster_m_block = tile_idx % (unsigned int)num_m_blocks;
                unsigned int bh = tile_idx / (unsigned int)num_m_blocks;
                unsigned int m_block = cluster_m_block * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                int head = bh % (unsigned int)heads;
                int batch_idx = bh / (unsigned int)heads;
                int causal_row = (m_block + stage) * (unsigned int)BLOCK_M + (unsigned int)(warp % 4 * 32 + lane);
                float row_max = -WAN_HYBRID_INF;
                float row_sum = 0.0f;
                #pragma unroll 1
                for (unsigned int n_iter = 0; n_iter < num_n_blocks; n_iter++) {
                    int n_block = num_n_blocks - 1 - n_iter;
                    mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    int s_addr = taddr + (unsigned int)TMEM_SCORES_OFFSET + stage * 128 + (unsigned int)(warp % 4 * 32 << 16);
                    float sv[128];
                    float tile_max = -WAN_HYBRID_INF;
                    float group_max0 = -WAN_HYBRID_INF;
                    float group_max1 = -WAN_HYBRID_INF;
                    float group_max2 = -WAN_HYBRID_INF;
                    float group_max3 = -WAN_HYBRID_INF;
                    float group_max4 = -WAN_HYBRID_INF;
                    float group_max5 = -WAN_HYBRID_INF;
                    float group_max6 = -WAN_HYBRID_INF;
                    float group_max7 = -WAN_HYBRID_INF;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63])
                        : "r"(s_addr)
                        : "memory");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95]), "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127])
                        : "r"(s_addr + 64)
                        : "memory");
                    float unused_group_max = -WAN_HYBRID_INF;
                    float2 _reg_reduce_max2_0 = {-WAN_HYBRID_INF, -WAN_HYBRID_INF};
                    row_max_x32_accum(&sv[0], _reg_reduce_max2_0);
                    row_max_x32_accum(&sv[32], _reg_reduce_max2_0);
                    row_max_x32_accum(&sv[64], _reg_reduce_max2_0);
                    row_max_x32_accum(&sv[96], _reg_reduce_max2_0);
                    float sv_max = row_max_reduce(_reg_reduce_max2_0);
                    tile_max = sv_max;
                    group_max0 = unused_group_max;
                    group_max1 = unused_group_max;
                    group_max2 = unused_group_max;
                    group_max3 = unused_group_max;
                    group_max4 = unused_group_max;
                    group_max5 = unused_group_max;
                    group_max6 = unused_group_max;
                    group_max7 = unused_group_max;
                    {
                    }
                    int tail_valid = seqlen_kv - n_block * BLOCK_N;
                    if (tail_valid < BLOCK_N) {
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_1 = tail_valid;
                            if (_lim_1 <= 0) { _slice_lo_mask_0 = 0u; }
                            else if (_lim_1 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_1));
                            }
                        }
                        #pragma unroll
                        for (int _i_2 = 0; _i_2 < 32; _i_2++) {
                            if (!(_slice_lo_mask_0 & (1u << _i_2))) sv[0 + _i_2] = -WAN_HYBRID_INF;
                        }
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_3 = tail_valid - 32;
                            if (_lim_3 <= 0) { _slice_lo_mask_1 = 0u; }
                            else if (_lim_3 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_3));
                            }
                        }
                        #pragma unroll
                        for (int _i_4 = 0; _i_4 < 32; _i_4++) {
                            if (!(_slice_lo_mask_1 & (1u << _i_4))) sv[32 + _i_4] = -WAN_HYBRID_INF;
                        }
                        uint32_t _slice_lo_mask_2;
                        {
                            int _lim_5 = tail_valid - 64;
                            if (_lim_5 <= 0) { _slice_lo_mask_2 = 0u; }
                            else if (_lim_5 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_5));
                            }
                        }
                        #pragma unroll
                        for (int _i_6 = 0; _i_6 < 32; _i_6++) {
                            if (!(_slice_lo_mask_2 & (1u << _i_6))) sv[64 + _i_6] = -WAN_HYBRID_INF;
                        }
                        uint32_t _slice_lo_mask_3;
                        {
                            int _lim_7 = tail_valid - 96;
                            if (_lim_7 <= 0) { _slice_lo_mask_3 = 0u; }
                            else if (_lim_7 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_7));
                            }
                        }
                        #pragma unroll
                        for (int _i_8 = 0; _i_8 < 32; _i_8++) {
                            if (!(_slice_lo_mask_3 & (1u << _i_8))) sv[96 + _i_8] = -WAN_HYBRID_INF;
                        }
                        float2 _reg_reduce_max2_9 = {-WAN_HYBRID_INF, -WAN_HYBRID_INF};
                        row_max_x32_accum(&sv[0], _reg_reduce_max2_9);
                        row_max_x32_accum(&sv[32], _reg_reduce_max2_9);
                        row_max_x32_accum(&sv[64], _reg_reduce_max2_9);
                        row_max_x32_accum(&sv[96], _reg_reduce_max2_9);
                        float sv_max_0 = row_max_reduce(_reg_reduce_max2_9);
                        tile_max = sv_max_0;
                    }
                    float _max_0 = max_noftz(tile_max, row_max);
                    float new_max = _max_0;
                    float safe_max = ((new_max == -WAN_HYBRID_INF) ? 0.0f : new_max);
                    float new_max_scaled = safe_max * softmax_scale_log2;
                    float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                    float acc_scale_log2 = _fma_0;
                    float acc_scale;
                    if (acc_scale_log2 >= -11.0f) {
                        safe_max = ((row_max == -WAN_HYBRID_INF) ? 0.0f : row_max);
                        acc_scale = 1.0f;
                        new_max_scaled = safe_max * softmax_scale_log2;
                    } else {
                        float _exp2_0 = approx_exp2(acc_scale_log2);
                        acc_scale = ((row_max > -WAN_HYBRID_INF) ? _exp2_0 : 1.0f);
                        row_max = new_max;
                    }
                    int skip_owner_rescale = 1;
                    {
                        int _vote_0 = __all_sync(0xFFFFFFFF, acc_scale == 1.0f);
                        skip_owner_rescale = _vote_0;
                    }
                    float sf_values[4];
                    {
                        float sv_max_0_1 = sv[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            sv_max_0_1 = max_noftz(sv_max_0_1, sv[_lr]);
                        }
                        float block_max = sv_max_0_1;
                        float block_max_scaled = ((block_max > -WAN_HYBRID_INF) ? block_max * softmax_scale_log2 : 0.0f);
                        float _exp2_1 = approx_exp2(block_max_scaled - new_max_scaled - 2.584962500721156f);
                        float p_scale = ((block_max > -WAN_HYBRID_INF) ? _exp2_1 : 0.0f);
                        sf_values[0] = p_scale;
                        const float2 _fma_b2_10 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_11 = {2.584962500721156f - block_max_scaled, 2.584962500721156f - block_max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 8; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_10, _fma_c2_11);
                        #pragma unroll
                        for (int _le = 0; _le < 16; _le++) {
                            sv[_le] = approx_exp2(sv[_le]);
                        }
                        float sv_max_1 = sv[(16) + 0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            sv_max_1 = max_noftz(sv_max_1, sv[(16) + _lr]);
                        }
                        float block_max_2 = sv_max_1;
                        float block_max_scaled_3 = ((block_max_2 > -WAN_HYBRID_INF) ? block_max_2 * softmax_scale_log2 : 0.0f);
                        float _exp2_2 = approx_exp2(block_max_scaled_3 - new_max_scaled - 2.584962500721156f);
                        float p_scale_4 = ((block_max_2 > -WAN_HYBRID_INF) ? _exp2_2 : 0.0f);
                        sf_values[1] = p_scale_4;
                        const float2 _fma_b2_12 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_13 = {2.584962500721156f - block_max_scaled_3, 2.584962500721156f - block_max_scaled_3};
                        #pragma unroll
                        for (int _lf = 0; _lf < 8; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_12, _fma_c2_13);
                        #pragma unroll
                        for (int _le = 0; _le < 16; _le++) {
                            sv[_le + 16] = approx_exp2(sv[_le + 16]);
                        }
                        float sv_max_5 = sv[(32) + 0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            sv_max_5 = max_noftz(sv_max_5, sv[(32) + _lr]);
                        }
                        float block_max_6 = sv_max_5;
                        float block_max_scaled_7 = ((block_max_6 > -WAN_HYBRID_INF) ? block_max_6 * softmax_scale_log2 : 0.0f);
                        float _exp2_3 = approx_exp2(block_max_scaled_7 - new_max_scaled - 2.584962500721156f);
                        float p_scale_8 = ((block_max_6 > -WAN_HYBRID_INF) ? _exp2_3 : 0.0f);
                        sf_values[2] = p_scale_8;
                        const float2 _fma_b2_14 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_15 = {2.584962500721156f - block_max_scaled_7, 2.584962500721156f - block_max_scaled_7};
                        #pragma unroll
                        for (int _lf = 0; _lf < 8; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 32))[_lf], _fma_b2_14, _fma_c2_15);
                        #pragma unroll
                        for (int _le = 0; _le < 16; _le++) {
                            sv[_le + 32] = approx_exp2(sv[_le + 32]);
                        }
                        float sv_max_9 = sv[(48) + 0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            sv_max_9 = max_noftz(sv_max_9, sv[(48) + _lr]);
                        }
                        float block_max_10 = sv_max_9;
                        float block_max_scaled_11 = ((block_max_10 > -WAN_HYBRID_INF) ? block_max_10 * softmax_scale_log2 : 0.0f);
                        float _exp2_4 = approx_exp2(block_max_scaled_11 - new_max_scaled - 2.584962500721156f);
                        float p_scale_12 = ((block_max_10 > -WAN_HYBRID_INF) ? _exp2_4 : 0.0f);
                        sf_values[3] = p_scale_12;
                        const float2 _fma_b2_16 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_17 = {2.584962500721156f - block_max_scaled_11, 2.584962500721156f - block_max_scaled_11};
                        #pragma unroll
                        for (int _lf = 0; _lf < 8; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 48))[_lf], _fma_b2_16, _fma_c2_17);
                        #pragma unroll
                        for (int _le = 0; _le < 16; _le++) {
                            sv[_le + 48] = approx_exp2(sv[_le + 48]);
                        }
                    }
                    {
                        if (skip_owner_rescale == 0) {
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int owner_output_off = ((stage == 0) ? TMEM_OUTPUT_0_OFFSET : TMEM_OUTPUT_1_OFFSET);
                            #pragma unroll
                            for (int owner_col = 0; owner_col < HEAD_DIM / 16; owner_col++) {
                                int owner_output_addr = taddr + (unsigned int)owner_output_off + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(owner_col * 16);
                                float _tmem_load_0[16];
                                tmem_ld_x16(&_tmem_load_0[0], owner_output_addr);
                                const float2 _scale2_18 = {acc_scale, acc_scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_18);
                                tmem_st_x16_f32(owner_output_addr, _tmem_load_0);
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                        }
                    }
                    float2 _f2_8 = make_float2(0.0f, 0.0f);
                    float2 block_sum2 = _f2_8;
                    float sf_residual_values[4];
                    float decoded_probability_values[16];
                    int row_outer = (warp % 4 * 32 + lane) / 32;
                    int row_lane = (warp % 4 * 32 + lane) % 32;
                    int row_group = row_lane / 8;
                    int row_in_group = row_lane % 8;
                    int sf_dst = (row_group * 8 + row_in_group) * 4 + row_outer;
                    {
                        float p_scale_1 = sf_values[0];
                        {
                            float2 _f2_9 = make_float2(sv[0], sv[1]);
                            float2 partial = _f2_9;
                            #pragma unroll
                            for (int pair = 2; pair < 16; pair += 2) {
                                float2 _f2_10 = make_float2((sv + 0)[pair], (sv + 0)[pair + 1]);
                                partial = add_f32x2(partial, _f2_10);
                            }
                            float2 frag_sum2 = partial;
                            float2 _f2_11 = make_float2(p_scale_1, p_scale_1);
                            float2 raw_scale2 = _f2_11;
                            float2 raw_sum2 = mul_f32x2(frag_sum2, raw_scale2);
                            uint32_t _fp4_0[2];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(sv[0]), "f"(sv[1]), "f"(sv[2]), "f"(sv[3]), "f"(sv[4]), "f"(sv[5]), "f"(sv[6]), "f"(sv[7]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[1]) : "f"(sv[8]), "f"(sv[9]), "f"(sv[10]), "f"(sv[11]), "f"(sv[12]), "f"(sv[13]), "f"(sv[14]), "f"(sv[15]));
                            block_sum2 = add_f32x2(block_sum2, raw_sum2);
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + (unsigned int)TMEM_SCORES_OFFSET + stage * 128 + (unsigned int)(warp % 4 * 32 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_0[1]))
                                : "memory");
                        }
                    }
                    {
                        float p_scale_2 = sf_values[1];
                        {
                            float2 _f2_31 = make_float2(sv[16], sv[17]);
                            float2 partial_1 = _f2_31;
                            #pragma unroll
                            for (int pair_1 = 2; pair_1 < 16; pair_1 += 2) {
                                float2 _f2_32 = make_float2((sv + 16)[pair_1], (sv + 16)[pair_1 + 1]);
                                partial_1 = add_f32x2(partial_1, _f2_32);
                            }
                            float2 frag_sum2_1 = partial_1;
                            float2 _f2_33 = make_float2(p_scale_2, p_scale_2);
                            float2 raw_scale2_1 = _f2_33;
                            float2 raw_sum2_1 = mul_f32x2(frag_sum2_1, raw_scale2_1);
                            uint32_t _fp4_4[2];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_4[0]) : "f"(sv[16]), "f"(sv[17]), "f"(sv[18]), "f"(sv[19]), "f"(sv[20]), "f"(sv[21]), "f"(sv[22]), "f"(sv[23]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_4[1]) : "f"(sv[24]), "f"(sv[25]), "f"(sv[26]), "f"(sv[27]), "f"(sv[28]), "f"(sv[29]), "f"(sv[30]), "f"(sv[31]));
                            block_sum2 = add_f32x2(block_sum2, raw_sum2_1);
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + (unsigned int)TMEM_SCORES_OFFSET + stage * 128 + (unsigned int)(warp % 4 * 32 << 16) + 2), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_4[1]))
                                : "memory");
                        }
                    }
                    {
                        float p_scale_3 = sf_values[2];
                        {
                            float2 _f2_53 = make_float2(sv[32], sv[33]);
                            float2 partial_2 = _f2_53;
                            #pragma unroll
                            for (int pair_2 = 2; pair_2 < 16; pair_2 += 2) {
                                float2 _f2_54 = make_float2((sv + 32)[pair_2], (sv + 32)[pair_2 + 1]);
                                partial_2 = add_f32x2(partial_2, _f2_54);
                            }
                            float2 frag_sum2_2 = partial_2;
                            float2 _f2_55 = make_float2(p_scale_3, p_scale_3);
                            float2 raw_scale2_2 = _f2_55;
                            float2 raw_sum2_2 = mul_f32x2(frag_sum2_2, raw_scale2_2);
                            uint32_t _fp4_8[2];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_8[0]) : "f"(sv[32]), "f"(sv[33]), "f"(sv[34]), "f"(sv[35]), "f"(sv[36]), "f"(sv[37]), "f"(sv[38]), "f"(sv[39]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_8[1]) : "f"(sv[40]), "f"(sv[41]), "f"(sv[42]), "f"(sv[43]), "f"(sv[44]), "f"(sv[45]), "f"(sv[46]), "f"(sv[47]));
                            block_sum2 = add_f32x2(block_sum2, raw_sum2_2);
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + (unsigned int)TMEM_SCORES_OFFSET + stage * 128 + (unsigned int)(warp % 4 * 32 << 16) + 4), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_8[1]))
                                : "memory");
                        }
                    }
                    {
                        float p_scale_5 = sf_values[3];
                        {
                            float2 _f2_75 = make_float2(sv[48], sv[49]);
                            float2 partial_3 = _f2_75;
                            #pragma unroll
                            for (int pair_3 = 2; pair_3 < 16; pair_3 += 2) {
                                float2 _f2_76 = make_float2((sv + 48)[pair_3], (sv + 48)[pair_3 + 1]);
                                partial_3 = add_f32x2(partial_3, _f2_76);
                            }
                            float2 frag_sum2_3 = partial_3;
                            float2 _f2_77 = make_float2(p_scale_5, p_scale_5);
                            float2 raw_scale2_3 = _f2_77;
                            float2 raw_sum2_3 = mul_f32x2(frag_sum2_3, raw_scale2_3);
                            uint32_t _fp4_12[2];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_12[0]) : "f"(sv[48]), "f"(sv[49]), "f"(sv[50]), "f"(sv[51]), "f"(sv[52]), "f"(sv[53]), "f"(sv[54]), "f"(sv[55]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_12[1]) : "f"(sv[56]), "f"(sv[57]), "f"(sv[58]), "f"(sv[59]), "f"(sv[60]), "f"(sv[61]), "f"(sv[62]), "f"(sv[63]));
                            block_sum2 = add_f32x2(block_sum2, raw_sum2_3);
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + (unsigned int)TMEM_SCORES_OFFSET + stage * 128 + (unsigned int)(warp % 4 * 32 << 16) + 6), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_12[1]))
                                : "memory");
                        }
                    }
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
                    int pv_sfa_lo_col = ((stage == 0) ? TMEM_TMEM_SFA_PV0_LO_OFFSET : TMEM_TMEM_SFA_PV1_LO_OFFSET) + warp % 4;
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)pv_sfa_lo_col), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_0[0]))
                        : "memory");
                    {
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    float sv_max_0_2 = sv[(64) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv_max_0_2 = max_noftz(sv_max_0_2, sv[(64) + _lr]);
                    }
                    float _exp2_9 = approx_exp2(((sv_max_0_2 > -WAN_HYBRID_INF) ? sv_max_0_2 * softmax_scale_log2 : 0.0f) - new_max_scaled - 2.584962500721156f);
                    sf_values[0] = ((sv_max_0_2 > -WAN_HYBRID_INF) ? _exp2_9 : 0.0f);
                    const float2 _fma_b2_19 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_20 = {2.584962500721156f - ((sv_max_0_2 > -WAN_HYBRID_INF) ? sv_max_0_2 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((sv_max_0_2 > -WAN_HYBRID_INF) ? sv_max_0_2 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 64))[_lf], _fma_b2_19, _fma_c2_20);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 64] = approx_exp2(sv[_le + 64]);
                    }
                    {
                        {
                            float2 _f2_97 = make_float2(sv[64], sv[65]);
                            float2 partial_4 = _f2_97;
                            #pragma unroll
                            for (int pair_4 = 2; pair_4 < 16; pair_4 += 2) {
                                float2 _f2_98 = make_float2((sv + 64)[pair_4], (sv + 64)[pair_4 + 1]);
                                partial_4 = add_f32x2(partial_4, _f2_98);
                            }
                            float2 frag_sum2_4 = partial_4;
                            float2 _f2_99 = make_float2(((sv_max_0_2 > -WAN_HYBRID_INF) ? _exp2_9 : 0.0f), ((sv_max_0_2 > -WAN_HYBRID_INF) ? _exp2_9 : 0.0f));
                            float2 raw_scale2_4 = _f2_99;
                            float2 raw_sum2_4 = mul_f32x2(frag_sum2_4, raw_scale2_4);
                            uint32_t _fp4_16[2];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_16[0]) : "f"(sv[64]), "f"(sv[65]), "f"(sv[66]), "f"(sv[67]), "f"(sv[68]), "f"(sv[69]), "f"(sv[70]), "f"(sv[71]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_16[1]) : "f"(sv[72]), "f"(sv[73]), "f"(sv[74]), "f"(sv[75]), "f"(sv[76]), "f"(sv[77]), "f"(sv[78]), "f"(sv[79]));
                            block_sum2 = add_f32x2(block_sum2, raw_sum2_4);
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + (unsigned int)TMEM_SCORES_OFFSET + stage * 128 + (unsigned int)(warp % 4 * 32 << 16) + 8), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_16[1]))
                                : "memory");
                        }
                    }
                    float sv_max_1_1 = sv[(80) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv_max_1_1 = max_noftz(sv_max_1_1, sv[(80) + _lr]);
                    }
                    float _exp2_10 = approx_exp2(((sv_max_1_1 > -WAN_HYBRID_INF) ? sv_max_1_1 * softmax_scale_log2 : 0.0f) - new_max_scaled - 2.584962500721156f);
                    sf_values[1] = ((sv_max_1_1 > -WAN_HYBRID_INF) ? _exp2_10 : 0.0f);
                    const float2 _fma_b2_21 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_22 = {2.584962500721156f - ((sv_max_1_1 > -WAN_HYBRID_INF) ? sv_max_1_1 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((sv_max_1_1 > -WAN_HYBRID_INF) ? sv_max_1_1 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 80))[_lf], _fma_b2_21, _fma_c2_22);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 80] = approx_exp2(sv[_le + 80]);
                    }
                    {
                        {
                            float2 _f2_119 = make_float2(sv[80], sv[81]);
                            float2 partial_5 = _f2_119;
                            #pragma unroll
                            for (int pair_5 = 2; pair_5 < 16; pair_5 += 2) {
                                float2 _f2_120 = make_float2((sv + 80)[pair_5], (sv + 80)[pair_5 + 1]);
                                partial_5 = add_f32x2(partial_5, _f2_120);
                            }
                            float2 frag_sum2_5 = partial_5;
                            float2 _f2_121 = make_float2(((sv_max_1_1 > -WAN_HYBRID_INF) ? _exp2_10 : 0.0f), ((sv_max_1_1 > -WAN_HYBRID_INF) ? _exp2_10 : 0.0f));
                            float2 raw_scale2_5 = _f2_121;
                            float2 raw_sum2_5 = mul_f32x2(frag_sum2_5, raw_scale2_5);
                            uint32_t _fp4_20[2];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_20[0]) : "f"(sv[80]), "f"(sv[81]), "f"(sv[82]), "f"(sv[83]), "f"(sv[84]), "f"(sv[85]), "f"(sv[86]), "f"(sv[87]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_20[1]) : "f"(sv[88]), "f"(sv[89]), "f"(sv[90]), "f"(sv[91]), "f"(sv[92]), "f"(sv[93]), "f"(sv[94]), "f"(sv[95]));
                            block_sum2 = add_f32x2(block_sum2, raw_sum2_5);
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + (unsigned int)TMEM_SCORES_OFFSET + stage * 128 + (unsigned int)(warp % 4 * 32 << 16) + 10), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_20[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_20[1]))
                                : "memory");
                        }
                    }
                    float sv_max_2 = sv[(96) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv_max_2 = max_noftz(sv_max_2, sv[(96) + _lr]);
                    }
                    float _exp2_11 = approx_exp2(((sv_max_2 > -WAN_HYBRID_INF) ? sv_max_2 * softmax_scale_log2 : 0.0f) - new_max_scaled - 2.584962500721156f);
                    sf_values[2] = ((sv_max_2 > -WAN_HYBRID_INF) ? _exp2_11 : 0.0f);
                    const float2 _fma_b2_23 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_24 = {2.584962500721156f - ((sv_max_2 > -WAN_HYBRID_INF) ? sv_max_2 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((sv_max_2 > -WAN_HYBRID_INF) ? sv_max_2 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 96))[_lf], _fma_b2_23, _fma_c2_24);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 96] = approx_exp2(sv[_le + 96]);
                    }
                    {
                        {
                            float2 _f2_141 = make_float2(sv[96], sv[97]);
                            float2 partial_6 = _f2_141;
                            #pragma unroll
                            for (int pair_6 = 2; pair_6 < 16; pair_6 += 2) {
                                float2 _f2_142 = make_float2((sv + 96)[pair_6], (sv + 96)[pair_6 + 1]);
                                partial_6 = add_f32x2(partial_6, _f2_142);
                            }
                            float2 frag_sum2_6 = partial_6;
                            float2 _f2_143 = make_float2(((sv_max_2 > -WAN_HYBRID_INF) ? _exp2_11 : 0.0f), ((sv_max_2 > -WAN_HYBRID_INF) ? _exp2_11 : 0.0f));
                            float2 raw_scale2_6 = _f2_143;
                            float2 raw_sum2_6 = mul_f32x2(frag_sum2_6, raw_scale2_6);
                            uint32_t _fp4_24[2];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_24[0]) : "f"(sv[96]), "f"(sv[97]), "f"(sv[98]), "f"(sv[99]), "f"(sv[100]), "f"(sv[101]), "f"(sv[102]), "f"(sv[103]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_24[1]) : "f"(sv[104]), "f"(sv[105]), "f"(sv[106]), "f"(sv[107]), "f"(sv[108]), "f"(sv[109]), "f"(sv[110]), "f"(sv[111]));
                            block_sum2 = add_f32x2(block_sum2, raw_sum2_6);
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + (unsigned int)TMEM_SCORES_OFFSET + stage * 128 + (unsigned int)(warp % 4 * 32 << 16) + 12), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_24[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_24[1]))
                                : "memory");
                        }
                    }
                    float sv_max_3 = sv[(112) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        sv_max_3 = max_noftz(sv_max_3, sv[(112) + _lr]);
                    }
                    float _exp2_12 = approx_exp2(((sv_max_3 > -WAN_HYBRID_INF) ? sv_max_3 * softmax_scale_log2 : 0.0f) - new_max_scaled - 2.584962500721156f);
                    sf_values[3] = ((sv_max_3 > -WAN_HYBRID_INF) ? _exp2_12 : 0.0f);
                    const float2 _fma_b2_25 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_26 = {2.584962500721156f - ((sv_max_3 > -WAN_HYBRID_INF) ? sv_max_3 * softmax_scale_log2 : 0.0f), 2.584962500721156f - ((sv_max_3 > -WAN_HYBRID_INF) ? sv_max_3 * softmax_scale_log2 : 0.0f)};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 112))[_lf], _fma_b2_25, _fma_c2_26);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 112] = approx_exp2(sv[_le + 112]);
                    }
                    {
                        {
                            float2 _f2_163 = make_float2(sv[112], sv[113]);
                            float2 partial_7 = _f2_163;
                            #pragma unroll
                            for (int pair_7 = 2; pair_7 < 16; pair_7 += 2) {
                                float2 _f2_164 = make_float2((sv + 112)[pair_7], (sv + 112)[pair_7 + 1]);
                                partial_7 = add_f32x2(partial_7, _f2_164);
                            }
                            float2 frag_sum2_7 = partial_7;
                            float2 _f2_165 = make_float2(((sv_max_3 > -WAN_HYBRID_INF) ? _exp2_12 : 0.0f), ((sv_max_3 > -WAN_HYBRID_INF) ? _exp2_12 : 0.0f));
                            float2 raw_scale2_7 = _f2_165;
                            float2 raw_sum2_7 = mul_f32x2(frag_sum2_7, raw_scale2_7);
                            uint32_t _fp4_28[2];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_28[0]) : "f"(sv[112]), "f"(sv[113]), "f"(sv[114]), "f"(sv[115]), "f"(sv[116]), "f"(sv[117]), "f"(sv[118]), "f"(sv[119]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_28[1]) : "f"(sv[120]), "f"(sv[121]), "f"(sv[122]), "f"(sv[123]), "f"(sv[124]), "f"(sv[125]), "f"(sv[126]), "f"(sv[127]));
                            block_sum2 = add_f32x2(block_sum2, raw_sum2_7);
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + (unsigned int)TMEM_SCORES_OFFSET + stage * 128 + (unsigned int)(warp % 4 * 32 << 16) + 14), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_28[0])), "r"(*reinterpret_cast<const uint32_t*>(&_fp4_28[1]))
                                : "memory");
                        }
                    }
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
                            : "=r"(_packed) : "f"(sf_values[0]), "f"(sf_values[1]),
                                               "f"(sf_values[2]), "f"(sf_values[3]));
                        _fp8_2[0] = _packed;
                    }
                    int pv_sfa_hi_col = ((stage == 0) ? TMEM_TMEM_SFA_PV0_HI_OFFSET : TMEM_TMEM_SFA_PV1_HI_OFFSET) + warp % 4;
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)pv_sfa_hi_col), "r"(*reinterpret_cast<const uint32_t*>(&_fp8_2[0]))
                        : "memory");
                    {
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_2_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    float block_sum = block_sum2.x + block_sum2.y;
                    row_sum = row_sum * acc_scale + block_sum;
                }
                mbarrier_wait(o_full_addr + (stage) * 8, _phase_o_full);
                _phase_o_full ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _rcp_8 = approx_rcp(row_sum);
                float final_scale = ((row_sum != 0.0f && row_sum == row_sum) ? _rcp_8 : 0.0f);
                if (stage == 0) {
                    #pragma unroll
                    for (int col = 0; col < HEAD_DIM / 32; col++) {
                        int addr = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col * 16);
                        float _tmem_load_1[16];
                        tmem_ld_x16(&_tmem_load_1[0], addr);
                        const float2 _scale2_27 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_27);
                        uint32_t _tmem_load_1_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                            _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + col * 32 ^ ((warp % 4 * 32 + lane) * 128 + col * 32 >> 7 & 7) << 4))), "r"(_tmem_load_1_bf16[0]), "r"(_tmem_load_1_bf16[1]), "r"(_tmem_load_1_bf16[2]), "r"(_tmem_load_1_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col * 32 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col * 32 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_1_bf16[4]), "r"(_tmem_load_1_bf16[5]), "r"(_tmem_load_1_bf16[6]), "r"(_tmem_load_1_bf16[7]) : "memory");
                    }
                    #pragma unroll
                    for (int col_1 = 0; col_1 < HEAD_DIM / 32; col_1++) {
                        int addr_1 = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(HEAD_DIM / 2) + (unsigned int)(col_1 * 16);
                        float _tmem_load_2[16];
                        tmem_ld_x16(&_tmem_load_2[0], addr_1);
                        const float2 _scale2_28 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_28);
                        uint32_t _tmem_load_2_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                            _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + col_1 * 32 ^ ((warp % 4 * 32 + lane) * 128 + col_1 * 32 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[0]), "r"(_tmem_load_2_bf16[1]), "r"(_tmem_load_2_bf16[2]), "r"(_tmem_load_2_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_1 * 32 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_1 * 32 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[4]), "r"(_tmem_load_2_bf16[5]), "r"(_tmem_load_2_bf16[6]), "r"(_tmem_load_2_bf16[7]) : "memory");
                    }
                    asm volatile("fence.proxy.async;");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    int out_row0 = m_block * (unsigned int)BLOCK_M;
                    if (warp == 0) {
                        if (elect_sync()) {
                            tma_store_5d(O, 0, out_row0, head, batch_idx, 0, smem_o00_addr);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                } else {
                    #pragma unroll
                    for (int col_2 = 0; col_2 < HEAD_DIM / 32; col_2++) {
                        int addr_2 = taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_2 * 16);
                        float _tmem_load_3[16];
                        tmem_ld_x16(&_tmem_load_3[0], addr_2);
                        const float2 _scale2_29 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_29);
                        uint32_t _tmem_load_3_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                            _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + col_2 * 32 ^ ((warp % 4 * 32 + lane) * 128 + col_2 * 32 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[0]), "r"(_tmem_load_3_bf16[1]), "r"(_tmem_load_3_bf16[2]), "r"(_tmem_load_3_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_2 * 32 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_2 * 32 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[4]), "r"(_tmem_load_3_bf16[5]), "r"(_tmem_load_3_bf16[6]), "r"(_tmem_load_3_bf16[7]) : "memory");
                    }
                    #pragma unroll
                    for (int col_3 = 0; col_3 < HEAD_DIM / 32; col_3++) {
                        int addr_3 = taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(HEAD_DIM / 2) + (unsigned int)(col_3 * 16);
                        float _tmem_load_4[16];
                        tmem_ld_x16(&_tmem_load_4[0], addr_3);
                        const float2 _scale2_30 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_30);
                        uint32_t _tmem_load_4_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                            _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + col_3 * 32 ^ ((warp % 4 * 32 + lane) * 128 + col_3 * 32 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[0]), "r"(_tmem_load_4_bf16[1]), "r"(_tmem_load_4_bf16[2]), "r"(_tmem_load_4_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_3 * 32 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_3 * 32 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[4]), "r"(_tmem_load_4_bf16[5]), "r"(_tmem_load_4_bf16[6]), "r"(_tmem_load_4_bf16[7]) : "memory");
                    }
                    asm volatile("fence.proxy.async;");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int out_row1 = (m_block + 1) * (unsigned int)BLOCK_M;
                    if (warp == 4) {
                        if (elect_sync()) {
                            tma_store_5d(O, 0, out_row1, head, batch_idx, 0, smem_o10_addr);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                }
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((o_empty_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                mbarrier_arrive(tile_done_addr);
                if (warp == 0) {
                    mbarrier_wait(tile_done_addr, _phase_tile_done_0);
                    _phase_tile_done_0 ^= 1;
                    if (elect_sync()) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(peer_tile_done_addr), "r"(0) : "memory");
                    }
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // correction_main
            unsigned int total_tiles_1 = num_m_blocks * total_bh;
            int warp_row = warp % 4 * 32;
            int row = warp_row + lane;
            int row_base = warp_row << 16;
            unsigned int first_score_tile = 1;
            unsigned int correction_total_tiles = 0;
            unsigned int _phase_tile_start_0_1 = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_q_empty_1 = 1;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = cluster_id; tile_idx_1 < correction_total_tiles; tile_idx_1 += num_clusters) {
                unsigned int cluster_m_block_1 = tile_idx_1 % (unsigned int)num_m_blocks;
                unsigned int bh_1 = tile_idx_1 / (unsigned int)num_m_blocks;
                unsigned int m_block_1 = cluster_m_block_1 * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks_1 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                unsigned int correction_num_n_blocks = 1;
                #pragma unroll 1
                for (unsigned int n_iter_1 = 1; n_iter_1 < correction_num_n_blocks; n_iter_1++) {
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            unsigned int _phase_peer_tile_done_0 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_corr_done_0 = 0;
            unsigned int _phase_q_empty_0_1 = 1;
            unsigned int _phase_corr_done_1 = 0;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_o_empty_1 = 1;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_2_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_p_full_2_1 = 0;
            unsigned int _phase_pv_reads_done_0 = 0;
            unsigned int _phase_q_empty_1_1 = 1;
            if (cta_rank == 0) {
                unsigned int total_tiles_2 = num_m_blocks * total_bh;
                unsigned int k_stage = 0;
                unsigned int k_phase = 0;
                unsigned int v_stage = 0;
                unsigned int v_phase = 0;
                unsigned int first_tile = 1;
                #pragma unroll 1
                for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < total_tiles_2; tile_idx_2 += num_clusters) {
                    unsigned int cluster_m_block_2 = tile_idx_2 % (unsigned int)num_m_blocks;
                    unsigned int bh_2 = tile_idx_2 / (unsigned int)num_m_blocks;
                    unsigned int m_block_2 = cluster_m_block_2 * 4 + (unsigned int)(cta_rank * 2);
                    unsigned int num_n_blocks_2 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                    if (first_tile == 0) {
                        mbarrier_wait(peer_tile_done_addr, _phase_peer_tile_done_0);
                        _phase_peer_tile_done_0 ^= 1;
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(tile_start_addr);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(tile_start_addr), "r"(1) : "memory");
                    }
                    first_tile = 0;
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    mbarrier_wait(q_full_addr + 8, _phase_q_full_1);
                    _phase_q_full_1 ^= 1;
                    mbarrier_wait(k_full_addr + (k_stage) * 8, k_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 2048;
                    int _mma_b_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0));
                    elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                    int _mma_a_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 2048;
                    int _mma_b_lo_1 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_scores + (128))), "r"(0));
                    elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                    k_stage += 1;
                    if (k_stage == 2) { k_stage = 0; k_phase ^= 1; }
                    unsigned int first_pv = 1;
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                    _phase_o_empty_0 ^= 1;
                    mbarrier_wait(o_empty_addr + 8, _phase_o_empty_1);
                    _phase_o_empty_1 ^= 1;
                    #pragma unroll 1
                    for (unsigned int n_iter_2 = 0; n_iter_2 < num_n_blocks_2 - 1; n_iter_2++) {
                        int first_pv_flag = first_pv;
                        mbarrier_wait(v_full_addr + (v_stage) * 8, v_phase);
                        mbarrier_wait(k_full_addr + (k_stage) * 8, k_phase);
                        {
                            mbarrier_wait(p_full_addr, _phase_p_full_0);
                            _phase_p_full_0 ^= 1;
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_lo, make_sf_cp_desc_sbo128(smem_sfvt_lo_addr + v_stage * 1024));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_pv0_res_lo, make_sf_cp_desc_sbo128(smem_sfvt_residual_lo_addr + v_stage * 1024));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_hi, make_sf_cp_desc_sbo128(smem_sfvt_hi_addr + v_stage * 1024));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_pv0_res_hi, make_sf_cp_desc_sbo128(smem_sfvt_residual_hi_addr + v_stage * 1024));
                            }
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_2 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t b_desc = ((uint64_t)_mma_b_lo_2) | ((uint64_t)0x80004020 << 32);
                                    tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_0, tmem_scores + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag) ? 0 : 1));
                                }
                            }
                            int _mma_b_lo_3 = (((smem_vt_residual_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t b_desc = ((uint64_t)_mma_b_lo_3) | ((uint64_t)0x80004020 << 32);
                                    tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_0, tmem_scores + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfa_pv0_res_lo + 0, 1);
                                }
                            }
                            mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                            _phase_p_full_2_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_4 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t b_desc = ((uint64_t)_mma_b_lo_4) | ((uint64_t)0x80004020 << 32);
                                    tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_0, tmem_scores + 8 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                }
                            }
                            int _mma_b_lo_5 = (((smem_vt_residual_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t b_desc = ((uint64_t)_mma_b_lo_5) | ((uint64_t)0x80004020 << 32);
                                    tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_0, tmem_scores + 8 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfa_pv0_res_hi + 0, 1);
                                }
                            }
                            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                            _phase_p_full_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_6 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t b_desc = ((uint64_t)_mma_b_lo_6) | ((uint64_t)0x80004020 << 32);
                                    tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_1, tmem_scores + 128 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag) ? 0 : 1));
                                }
                            }
                            int _mma_b_lo_7 = (((smem_vt_residual_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t b_desc = ((uint64_t)_mma_b_lo_7) | ((uint64_t)0x80004020 << 32);
                                    tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_1, tmem_scores + 128 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfa_pv0_res_lo + 0, 1);
                                }
                            }
                            mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                            _phase_p_full_2_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_8 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t b_desc = ((uint64_t)_mma_b_lo_8) | ((uint64_t)0x80004020 << 32);
                                    tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_1, tmem_scores + 136 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                }
                            }
                            int _mma_b_lo_9 = (((smem_vt_residual_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t b_desc = ((uint64_t)_mma_b_lo_9) | ((uint64_t)0x80004020 << 32);
                                    tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_1, tmem_scores + 136 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfa_pv0_res_hi + 0, 1);
                                }
                            }
                            elect_commit_cg2_local(pv_reads_done_addr);
                            mbarrier_wait(pv_reads_done_addr, _phase_pv_reads_done_0);
                            _phase_pv_reads_done_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_10 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 2048;
                            int _mma_b_lo_10 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"(tmem_scores), "r"(0));
                            elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                            int _mma_a_lo_11 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 2048;
                            int _mma_b_lo_11 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"((tmem_scores + (128))), "r"(0));
                            elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                            k_stage += 1;
                            if (k_stage == 2) { k_stage = 0; k_phase ^= 1; }
                            elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                            v_stage += 1;
                            if (v_stage == 4) { v_stage = 0; v_phase ^= 1; }
                            first_pv = 0;
                        }
                    }
                    int first_pv_flag_1 = first_pv;
                    mbarrier_wait(v_full_addr + (v_stage) * 8, v_phase);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_lo, make_sf_cp_desc_sbo128(smem_sfvt_lo_addr + v_stage * 1024));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_pv0_res_lo, make_sf_cp_desc_sbo128(smem_sfvt_residual_lo_addr + v_stage * 1024));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_hi, make_sf_cp_desc_sbo128(smem_sfvt_hi_addr + v_stage * 1024));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_pv0_res_hi, make_sf_cp_desc_sbo128(smem_sfvt_residual_hi_addr + v_stage * 1024));
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_24 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_24) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_0, tmem_scores + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                        }
                    }
                    {
                        int _mma_b_lo_25 = (((smem_vt_residual_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_25) | ((uint64_t)0x80004020 << 32);
                                tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_0, tmem_scores + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfa_pv0_res_lo + 0, 1);
                            }
                        }
                    }
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                    _phase_p_full_2_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_26 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_26) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_0, tmem_scores + 8 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                        }
                    }
                    {
                        int _mma_b_lo_27 = (((smem_vt_residual_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_27) | ((uint64_t)0x80004020 << 32);
                                tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_0, tmem_scores + 8 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfa_pv0_res_hi + 0, 1);
                            }
                        }
                    }
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_28 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_28) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_1, tmem_scores + 128 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                        }
                    }
                    {
                        int _mma_b_lo_29 = (((smem_vt_residual_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_29) | ((uint64_t)0x80004020 << 32);
                                tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_1, tmem_scores + 128 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfa_pv0_res_lo + 0, 1);
                            }
                        }
                    }
                    mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                    _phase_p_full_2_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_30 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_30) | ((uint64_t)0x80004020 << 32);
                            tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_1, tmem_scores + 136 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                        }
                    }
                    {
                        int _mma_b_lo_31 = (((smem_vt_residual_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t b_desc = ((uint64_t)_mma_b_lo_31) | ((uint64_t)0x80004020 << 32);
                                tcgen05_mma_mxf4nvf4_bs_ts_cta2(tmem_output_1, tmem_scores + 136 + 0, b_desc + 0, 0x10200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfa_pv0_res_hi + 0, 1);
                            }
                        }
                    }
                    elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                    v_stage += 1;
                    if (v_stage == 4) { v_stage = 0; v_phase ^= 1; }
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            unsigned int total_tiles_3 = num_m_blocks * total_bh;
            unsigned int num_n_blocks_all = (seqlen_kv + 4 * BLOCK_N - 1) / (4 * BLOCK_N) * 4;
            unsigned int num_q_blocks_all = q_stride / BLOCK_M;
            unsigned int v_level_row_stride = total_bh * HEAD_DIM;
            unsigned int v_scale_level_stride = (unsigned int)total_bh * num_n_blocks_all * 16;
            unsigned int k_load_stage = 0;
            unsigned int v_load_stage = 0;
            unsigned int _phase_q_empty_0_2 = 1;
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < total_tiles_3; tile_idx_3 += num_clusters) {
                unsigned int cluster_m_block_3 = tile_idx_3 % (unsigned int)num_m_blocks;
                unsigned int bh_3 = tile_idx_3 / (unsigned int)num_m_blocks;
                unsigned int m_block_3 = cluster_m_block_3 * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks_3 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                int head_1 = bh_3 % (unsigned int)heads;
                int batch_idx_1 = bh_3 / (unsigned int)heads;
                int q_row = m_block_3 * (unsigned int)BLOCK_M;
                int q_sf_tile = bh_3 * num_q_blocks_all + m_block_3;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0_2);
                _phase_q_empty_0_2 ^= 1;
                if (elect_sync()) {
                    tma_5d_gmem2smem_cta2(smem_q_addr, Q, 0, q_row, head_1, batch_idx_1, 0, ((q_full_addr) & 0xFEFFFFFF));
                    tma_5d_gmem2smem_cta2(smem_q_addr + 32768, Q, 0, q_row + BLOCK_M, head_1, batch_idx_1, 0, ((q_full_addr + 8) & 0xFEFFFFFF));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr + 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                }
                #pragma unroll 1
                for (unsigned int ni = 0; ni < num_n_blocks_3; ni++) {
                    unsigned int n = num_n_blocks_3 - 1 - ni;
                    int kv_row = bh_3 * (unsigned int)kv_stride + n * (unsigned int)BLOCK_N;
                    int kv_sf_tile = bh_3 * num_n_blocks_all + n;
                    int vt_row = bh_3 * (unsigned int)HEAD_DIM;
                    int vt_col = n * (unsigned int)(BLOCK_N / 2);
                    mbarrier_wait(k_empty_addr + (k_load_stage) * 8, _phase_k_empty);
                    if (elect_sync()) {
                        tma_5d_gmem2smem_cta2(smem_k_addr + k_load_stage * 16384, K, 0, n * (unsigned int)BLOCK_N + (unsigned int)(cta_rank * 64), head_1, batch_idx_1, 0, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    }
                    k_load_stage += 1;
                    if (k_load_stage == 2) { k_load_stage = 0; _phase_k_empty ^= 1; }
                    mbarrier_wait(v_empty_addr + (v_load_stage) * 8, _phase_v_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_vt_addr + v_load_stage * 4096, Vt, vt_col, vt_row + cta_rank * 64, ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        {
                            tma_2d_gmem2smem_cta2(smem_vt_residual_addr + v_load_stage * 4096, Vt, vt_col, v_level_row_stride + (unsigned int)vt_row + (unsigned int)(cta_rank * 64), ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        }
                        tma_2d_gmem2smem_cta2(smem_sfvt_lo_addr + v_load_stage * 1024, SFVtLo, 0, kv_sf_tile * 16, ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        {
                            tma_2d_gmem2smem_cta2(smem_sfvt_residual_lo_addr + v_load_stage * 1024, SFVtLo, 0, v_scale_level_stride + (unsigned int)(kv_sf_tile * 16), ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        }
                        tma_2d_gmem2smem_cta2(smem_sfvt_hi_addr + v_load_stage * 1024, SFVtHi, 0, kv_sf_tile * 16, ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        {
                            tma_2d_gmem2smem_cta2(smem_sfvt_residual_hi_addr + v_load_stage * 1024, SFVtHi, 0, v_scale_level_stride + (unsigned int)(kv_sf_tile * 16), ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        }
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(10240)) : "memory");
                    }
                    v_load_stage += 1;
                    if (v_load_stage == 4) { v_load_stage = 0; _phase_v_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: _idle ----
    if (warp >= 14 && warp <= 15) {
        // idle — no tasks assigned
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
