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

// Frozen generated FlashKDA backward bundle for exact SM103a.
// Raw generated body SHA256: a2e40e9e246f1fe5d9fbef1cee469293936821cc3e5b15f315a554b01f55d5c5
// Normalized generated SHA256: 456a7ffd07e71c4d33d0257ca64ed6993e4aa160b69a6b93f60535f5b4de4984
// clang-format off
// BEGIN FROZEN GENERATED BODY
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) FlashKDATensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) FlashKDATensorMapPack { FlashKDATensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

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
           "r"(i_desc), "r"(enable_input_d));
}

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ void mma_ts_step(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
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

__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
}

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}

__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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

__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}

__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

#define FLASHKDA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define D 128

extern "C" {

__global__ __launch_bounds__(32) void
kernel_flashkda_backward_preprocess(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ beta, float* __restrict__ A_log, float* __restrict__ dt_bias, float* __restrict__ q_norm, float* __restrict__ k_norm, float* __restrict__ decay, float* __restrict__ beta_active, int total_tokens, int num_heads, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int token = blockIdx.x;
    int head = blockIdx.y;
    int elem = lane * 4;
    long long base = ((long long)token * (long long)num_heads + (long long)head) * (long long)D;
    float q_frag[4];
    float k_frag[4];
    float g_frag[4];
    {
        uint2 _vld_0;
        _vld_0 = *reinterpret_cast<const uint2*>(q + base + (long long)elem);
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&q_frag[0 + _pair * 2])[0]), "=f"((&q_frag[0 + _pair * 2])[1])
                : "r"(_vpairs_0[_pair]));
        }
    }
    {
        uint2 _vld_1;
        _vld_1 = *reinterpret_cast<const uint2*>(k + base + (long long)elem);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&k_frag[0 + _pair * 2])[0]), "=f"((&k_frag[0 + _pair * 2])[1])
                : "r"(_vpairs_1[_pair]));
        }
    }
    {
        uint2 _vld_2;
        _vld_2 = *reinterpret_cast<const uint2*>(g + base + (long long)elem);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&g_frag[0 + _pair * 2])[0]), "=f"((&g_frag[0 + _pair * 2])[1])
                : "r"(_vpairs_2[_pair]));
        }
    }
    float q_sq = 0.0f;
    float k_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float _fma_0 = __fmaf_rn(q_frag[i], q_frag[i], q_sq);
        q_sq = _fma_0;
        float _fma_1 = __fmaf_rn(k_frag[i], k_frag[i], k_sq);
        k_sq = _fma_1;
    }
    float _warp_reduce_0 = q_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    q_sq = _warp_reduce_0;
    float _warp_reduce_1 = k_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    k_sq = _warp_reduce_1;
    float _rsqrt_0 = rsqrtf(q_sq + 1e-06f);
    float q_inv = _rsqrt_0;
    float _rsqrt_1 = rsqrtf(k_sq + 1e-06f);
    float k_inv = _rsqrt_1;
    float _expf_0 = __expf(A_log[head]);
    float gate_a = _expf_0;
    #pragma unroll
    for (int i2 = 0; i2 < 4; i2++) {
        int dim = elem + i2;
        float biased = g_frag[i2] + dt_bias[head * D + dim];
        float _expf_1 = __expf((-gate_a) * biased);
        float gate_sigmoid = 1.0f / (1.0f + _expf_1);
        q_norm[base + (long long)dim] = q_frag[i2] * q_inv;
        k_norm[base + (long long)dim] = k_frag[i2] * k_inv;
        float _expf_2 = __expf(lower_bound * gate_sigmoid);
        decay[base + (long long)dim] = _expf_2;
    }
    if (lane == 0) {
        long long beta_index = (long long)token * (long long)num_heads + (long long)head;
        float beta_raw = (float)beta[beta_index];
        float _expf_3 = __expf(-beta_raw);
        beta_active[beta_index] = 1.0f / (1.0f + _expf_3);
    }
}

} // extern "C"

#undef D
#undef FLASHKDA_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHKDA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define D 128
#define V 128

extern "C" {

__global__ __launch_bounds__(32) void
kernel_flashkda_backward_checkpoint(float* __restrict__ k_norm, float* __restrict__ decay, float* __restrict__ beta_active, __nv_bfloat16* __restrict__ v, float* __restrict__ initial_state, long long* __restrict__ cu_seqlens, float* __restrict__ checkpoint, int num_sequences, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int work = blockIdx.x;
    int rows_per_sequence = num_heads * V;
    int sequence = work / rows_per_sequence;
    int remainder = work - sequence * rows_per_sequence;
    int head = remainder / V;
    int value_row = remainder - head * V;
    int elem = lane * 4;
    long long bos = cu_seqlens[sequence];
    long long eos = cu_seqlens[sequence + 1];
    long long state_base = (((long long)sequence * (long long)num_heads + (long long)head) * (long long)V + (long long)value_row) * (long long)D;
    float state[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(initial_state + state_base + (long long)elem);
        state[0 + 0] = _v4.x;
        state[0 + 1] = _v4.y;
        state[0 + 2] = _v4.z;
        state[0 + 3] = _v4.w;
    }
    #pragma unroll 1
    for (long long token = bos; token < eos; token++) {
        long long token_base = (token * (long long)num_heads + (long long)head) * (long long)D;
        float k_frag[4];
        float d_frag[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(k_norm + token_base + (long long)elem);
            k_frag[0 + 0] = _v4.x;
            k_frag[0 + 1] = _v4.y;
            k_frag[0 + 2] = _v4.z;
            k_frag[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(decay + token_base + (long long)elem);
            d_frag[0 + 0] = _v4.x;
            d_frag[0 + 1] = _v4.y;
            d_frag[0 + 2] = _v4.z;
            d_frag[0 + 3] = _v4.w;
        }
        float pred = 0.0f;
        float decayed[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            decayed[i] = state[i] * d_frag[i];
            float _fma_0 = __fmaf_rn(k_frag[i], decayed[i], pred);
            pred = _fma_0;
        }
        float _warp_reduce_0 = pred;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        pred = _warp_reduce_0;
        long long beta_index = token * (long long)num_heads + (long long)head;
        long long value_index = token_base + (long long)value_row;
        float residual = beta_active[beta_index] * ((float)v[value_index] - pred);
        long long checkpoint_base = ((token * (long long)num_heads + (long long)head) * (long long)V + (long long)value_row) * (long long)D;
        #pragma unroll
        for (int i2 = 0; i2 < 4; i2++) {
            float _fma_1 = __fmaf_rn(residual, k_frag[i2], decayed[i2]);
            state[i2] = _fma_1;
            checkpoint[checkpoint_base + (long long)elem + (long long)i2] = state[i2];
        }
    }
}

} // extern "C"

#undef D
#undef FLASHKDA_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef V

#define FLASHKDA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define D 128
#define V 128

extern "C" {

__global__ __launch_bounds__(32) void
kernel_flashkda_backward_reverse_rows(float* __restrict__ q_norm, float* __restrict__ k_norm, float* __restrict__ decay, float* __restrict__ beta_active, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ do_, float* __restrict__ initial_state, float* __restrict__ dfinal_state, long long* __restrict__ cu_seqlens, float* __restrict__ checkpoint, float* __restrict__ dq_normalized, float* __restrict__ dk_normalized, float* __restrict__ dlog_decay, float* __restrict__ dbeta_active, __nv_bfloat16* __restrict__ dv, float* __restrict__ dinitial_state, int num_sequences, int num_heads, float scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int work = blockIdx.x;
    int rows_per_sequence = num_heads * V;
    int sequence = work / rows_per_sequence;
    int remainder = work - sequence * rows_per_sequence;
    int head = remainder / V;
    int value_row = remainder - head * V;
    int elem = lane * 4;
    long long bos = cu_seqlens[sequence];
    long long eos = cu_seqlens[sequence + 1];
    long long sequence_length = eos - bos;
    long long state_base = (((long long)sequence * (long long)num_heads + (long long)head) * (long long)V + (long long)value_row) * (long long)D;
    float dstate[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(dfinal_state + state_base + (long long)elem);
        dstate[0 + 0] = _v4.x;
        dstate[0 + 1] = _v4.y;
        dstate[0 + 2] = _v4.z;
        dstate[0 + 3] = _v4.w;
    }
    #pragma unroll 1
    for (long long reverse_step = 0; reverse_step < sequence_length; reverse_step++) {
        long long token = eos - 1 - reverse_step;
        long long token_base = (token * (long long)num_heads + (long long)head) * (long long)D;
        long long checkpoint_base = ((token * (long long)num_heads + (long long)head) * (long long)V + (long long)value_row) * (long long)D;
        long long previous_base = checkpoint_base - (long long)num_heads * (long long)V * (long long)D;
        if (token == bos) {
            previous_base = state_base;
        }
        float q_frag[4];
        float k_frag[4];
        float d_frag[4];
        float state_now[4];
        float state_prev[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(q_norm + token_base + (long long)elem);
            q_frag[0 + 0] = _v4.x;
            q_frag[0 + 1] = _v4.y;
            q_frag[0 + 2] = _v4.z;
            q_frag[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(k_norm + token_base + (long long)elem);
            k_frag[0 + 0] = _v4.x;
            k_frag[0 + 1] = _v4.y;
            k_frag[0 + 2] = _v4.z;
            k_frag[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(decay + token_base + (long long)elem);
            d_frag[0 + 0] = _v4.x;
            d_frag[0 + 1] = _v4.y;
            d_frag[0 + 2] = _v4.z;
            d_frag[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(checkpoint + checkpoint_base + (long long)elem);
            state_now[0 + 0] = _v4.x;
            state_now[0 + 1] = _v4.y;
            state_now[0 + 2] = _v4.z;
            state_now[0 + 3] = _v4.w;
        }
        if (token == bos) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(initial_state + previous_base + (long long)elem);
                state_prev[0 + 0] = _v4.x;
                state_prev[0 + 1] = _v4.y;
                state_prev[0 + 2] = _v4.z;
                state_prev[0 + 3] = _v4.w;
            }
        } else {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(checkpoint + previous_base + (long long)elem);
                state_prev[0 + 0] = _v4.x;
                state_prev[0 + 1] = _v4.y;
                state_prev[0 + 2] = _v4.z;
                state_prev[0 + 3] = _v4.w;
            }
        }
        long long value_index = token_base + (long long)value_row;
        float output_adjoint = (float)do_[value_index];
        float pred = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float _fma_0 = __fmaf_rn(scale * output_adjoint, q_frag[i], dstate[i]);
            dstate[i] = _fma_0;
            float _fma_1 = __fmaf_rn(k_frag[i], state_prev[i] * d_frag[i], pred);
            pred = _fma_1;
        }
        float _warp_reduce_0 = pred;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        pred = _warp_reduce_0;
        float dr = 0.0f;
        #pragma unroll
        for (int i2 = 0; i2 < 4; i2++) {
            float _fma_2 = __fmaf_rn(dstate[i2], k_frag[i2], dr);
            dr = _fma_2;
        }
        float _warp_reduce_1 = dr;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        dr = _warp_reduce_1;
        long long beta_index = token * (long long)num_heads + (long long)head;
        float beta_value = beta_active[beta_index];
        float value_raw = (float)v[value_index];
        float residual = beta_value * (value_raw - pred);
        float dpred = (-dr) * beta_value;
        #pragma unroll
        for (int i3 = 0; i3 < 4; i3++) {
            long long dim_index = token_base + (long long)elem + (long long)i3;
            float decayed_state = state_prev[i3] * d_frag[i3];
            float _fma_3 = __fmaf_rn(dpred, k_frag[i3], dstate[i3]);
            float d_p = _fma_3;
            float _fma_4 = __fmaf_rn(dpred, decayed_state, residual * dstate[i3]);
            float d_k = _fma_4;
            atomicAdd(&dq_normalized[dim_index], scale * output_adjoint * state_now[i3]);
            atomicAdd(&dk_normalized[dim_index], d_k);
            atomicAdd(&dlog_decay[dim_index], d_p * state_prev[i3] * d_frag[i3]);
            dstate[i3] = d_p * d_frag[i3];
        }
        if (lane == 0) {
            atomicAdd(&dbeta_active[beta_index], dr * (value_raw - pred));
            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(dr * beta_value);
            dv[value_index] = _cvt_bf16_0;
        }
    }
    #pragma unroll
    for (int i4 = 0; i4 < 4; i4++) {
        dinitial_state[state_base + (long long)elem + (long long)i4] = dstate[i4];
    }
}

} // extern "C"

#undef D
#undef FLASHKDA_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef V

#define FLASHKDA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define D 128

extern "C" {

__global__ __launch_bounds__(32) void
kernel_flashkda_backward_finalize_tokens(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ g, float* __restrict__ beta_active, float* __restrict__ A_log, float* __restrict__ dt_bias, float* __restrict__ q_norm, float* __restrict__ k_norm, float* __restrict__ dq_normalized, float* __restrict__ dk_normalized, float* __restrict__ gate_common, float* __restrict__ dbeta_active, __nv_bfloat16* __restrict__ dq, __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dg, __nv_bfloat16* __restrict__ dbeta, int num_heads, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int token = blockIdx.x;
    int head = blockIdx.y;
    int elem = lane * 4;
    long long base = ((long long)token * (long long)num_heads + (long long)head) * (long long)D;
    float q_raw[4];
    float k_raw[4];
    float g_raw[4];
    float qn[4];
    float kn[4];
    float dqn[4];
    float dkn[4];
    float dlog[4];
    {
        uint2 _vld_0;
        _vld_0 = *reinterpret_cast<const uint2*>(q + base + (long long)elem);
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&q_raw[0 + _pair * 2])[0]), "=f"((&q_raw[0 + _pair * 2])[1])
                : "r"(_vpairs_0[_pair]));
        }
    }
    {
        uint2 _vld_1;
        _vld_1 = *reinterpret_cast<const uint2*>(k + base + (long long)elem);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&k_raw[0 + _pair * 2])[0]), "=f"((&k_raw[0 + _pair * 2])[1])
                : "r"(_vpairs_1[_pair]));
        }
    }
    {
        uint2 _vld_2;
        _vld_2 = *reinterpret_cast<const uint2*>(g + base + (long long)elem);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&g_raw[0 + _pair * 2])[0]), "=f"((&g_raw[0 + _pair * 2])[1])
                : "r"(_vpairs_2[_pair]));
        }
    }
    {
        float4 _v4 = *reinterpret_cast<const float4*>(q_norm + base + (long long)elem);
        qn[0 + 0] = _v4.x;
        qn[0 + 1] = _v4.y;
        qn[0 + 2] = _v4.z;
        qn[0 + 3] = _v4.w;
    }
    {
        float4 _v4 = *reinterpret_cast<const float4*>(k_norm + base + (long long)elem);
        kn[0 + 0] = _v4.x;
        kn[0 + 1] = _v4.y;
        kn[0 + 2] = _v4.z;
        kn[0 + 3] = _v4.w;
    }
    {
        float4 _v4 = *reinterpret_cast<const float4*>(dq_normalized + base + (long long)elem);
        dqn[0 + 0] = _v4.x;
        dqn[0 + 1] = _v4.y;
        dqn[0 + 2] = _v4.z;
        dqn[0 + 3] = _v4.w;
    }
    {
        float4 _v4 = *reinterpret_cast<const float4*>(dk_normalized + base + (long long)elem);
        dkn[0 + 0] = _v4.x;
        dkn[0 + 1] = _v4.y;
        dkn[0 + 2] = _v4.z;
        dkn[0 + 3] = _v4.w;
    }
    {
        float4 _v4 = *reinterpret_cast<const float4*>(gate_common + base + (long long)elem);
        dlog[0 + 0] = _v4.x;
        dlog[0 + 1] = _v4.y;
        dlog[0 + 2] = _v4.z;
        dlog[0 + 3] = _v4.w;
    }
    float q_sq = 0.0f;
    float k_sq = 0.0f;
    float q_dot = 0.0f;
    float k_dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float _fma_0 = __fmaf_rn(q_raw[i], q_raw[i], q_sq);
        q_sq = _fma_0;
        float _fma_1 = __fmaf_rn(k_raw[i], k_raw[i], k_sq);
        k_sq = _fma_1;
        float _fma_2 = __fmaf_rn(dqn[i], qn[i], q_dot);
        q_dot = _fma_2;
        float _fma_3 = __fmaf_rn(dkn[i], kn[i], k_dot);
        k_dot = _fma_3;
    }
    float _warp_reduce_0 = q_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    q_sq = _warp_reduce_0;
    float _warp_reduce_1 = k_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    k_sq = _warp_reduce_1;
    float _warp_reduce_2 = q_dot;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
    q_dot = _warp_reduce_2;
    float _warp_reduce_3 = k_dot;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
    k_dot = _warp_reduce_3;
    float _rsqrt_0 = rsqrtf(q_sq + 1e-06f);
    float q_inv = _rsqrt_0;
    float _rsqrt_1 = rsqrtf(k_sq + 1e-06f);
    float k_inv = _rsqrt_1;
    float _expf_0 = __expf(A_log[head]);
    float gate_a = _expf_0;
    #pragma unroll
    for (int i2 = 0; i2 < 4; i2++) {
        int dim = elem + i2;
        long long index = base + (long long)dim;
        float biased = g_raw[i2] + dt_bias[head * D + dim];
        float _expf_1 = __expf((-gate_a) * biased);
        float gate_sigmoid = 1.0f / (1.0f + _expf_1);
        float common = dlog[i2] * lower_bound * gate_sigmoid * (1.0f - gate_sigmoid) * gate_a;
        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(q_inv * (dqn[i2] - qn[i2] * q_dot));
        dq[index] = _cvt_bf16_0;
        __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(k_inv * (dkn[i2] - kn[i2] * k_dot));
        dk[index] = _cvt_bf16_1;
        __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(common);
        dg[index] = _cvt_bf16_2;
        gate_common[index] = common;
    }
    if (lane == 0) {
        long long beta_index = (long long)token * (long long)num_heads + (long long)head;
        float beta_sigmoid = beta_active[beta_index];
        __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(dbeta_active[beta_index] * beta_sigmoid * (1.0f - beta_sigmoid));
        dbeta[beta_index] = _cvt_bf16_3;
    }
}

} // extern "C"

#undef D
#undef FLASHKDA_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHKDA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_DA_OFF 0
#define SMEM_SMEM_DA_STAGE_BYTES 16
#define SMEM_SMEM_DA_STRIDE 16
#define SMEM_TOTAL 128
#define THREADS 128
#define D 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_flashkda_backward_gate_reduce_split(float* __restrict__ gate_common, __nv_bfloat16* __restrict__ g, float* __restrict__ dt_bias, float* __restrict__ dA_log, float* __restrict__ ddt_bias, int total_tokens, int num_heads, int tokens_per_split)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* smem_dA = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_dA_addr = smem + 0;

    // === Task calls (dependency order) ===
    int head = blockIdx.x;
    int split = blockIdx.y;
    int dim = tid;
    int token_start = split * tokens_per_split;
    int token_end = token_start + tokens_per_split;
    if (token_end > total_tokens) {
        token_end = total_tokens;
    }
    float bias = dt_bias[head * D + dim];
    float ddt_sum = 0.0f;
    float dA_sum = 0.0f;
    #pragma unroll 4
    for (int token = token_start; token < token_end; token++) {
        long long index = ((long long)token * (long long)num_heads + (long long)head) * (long long)D + (long long)dim;
        float common = gate_common[index];
        ddt_sum += common;
        float _fma_0 = __fmaf_rn(common, (float)g[index] + bias, dA_sum);
        dA_sum = _fma_0;
    }
    atomicAdd(&ddt_bias[head * D + dim], ddt_sum);
    float _warp_reduce_0 = dA_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    dA_sum = _warp_reduce_0;
    if (lane == 0) {
        smem_dA[warp] = dA_sum;
    }
    __syncthreads();
    if (warp == 0) {
        if (elect_sync()) {
            atomicAdd(&dA_log[head], smem_dA[0] + smem_dA[1] + smem_dA[2] + smem_dA[3]);
        }
    }
}

} // extern "C"

#undef D
#undef FLASHKDA_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DA_OFF
#undef SMEM_SMEM_DA_STAGE_BYTES
#undef SMEM_SMEM_DA_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef smem_dA_addr

#define FLASHKDA_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_TMEM_STATE_OFFSET 64
#define TMEM_TMEM_STATE_INP_OFFSET 0
#define TMEM_TMEM_U_ACC_OFFSET 224
#define TMEM_TMEM_U2_INP_OFFSET 224
#define TMEM_TMEM_U2_ACC_OFFSET 0
#define TMEM_TMEM_OUT_OFFSET 192
#define TMEM_TMEM_STATE_OUT_OFFSET 64
#define NUM_CHUNK_PIPE_STAGES 5
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 8192
#define SMEM_SMEM_QD_STRIDE 41984
#define SMEM_SMEM_G_RAW_OFF 1024
#define SMEM_SMEM_G_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_G_RAW_STRIDE 41984
#define SMEM_SMEM_G_RAW_ALL_OFF 1024
#define SMEM_SMEM_G_RAW_ALL_STAGE_BYTES 176128
#define SMEM_SMEM_G_RAW_ALL_STRIDE 176128
#define SMEM_SMEM_KD_OFF 9216
#define SMEM_SMEM_KD_STAGE_BYTES 8192
#define SMEM_SMEM_KD_STRIDE 41984
#define SMEM_SMEM_Q_RAW_PREFETCH_OFF 17408
#define SMEM_SMEM_Q_RAW_PREFETCH_STAGE_BYTES 8192
#define SMEM_SMEM_Q_RAW_PREFETCH_STRIDE 41984
#define SMEM_SMEM_FINAL_TRANS_OFF 17408
#define SMEM_SMEM_FINAL_TRANS_STAGE_BYTES 12288
#define SMEM_SMEM_FINAL_TRANS_STRIDE 41984
#define SMEM_SMEM_KR_TRANS_OFF 17408
#define SMEM_SMEM_KR_TRANS_STAGE_BYTES 8192
#define SMEM_SMEM_KR_TRANS_STRIDE 41984
#define SMEM_SMEM_MQK_TRANS_OFF 25600
#define SMEM_SMEM_MQK_TRANS_STAGE_BYTES 2048
#define SMEM_SMEM_MQK_TRANS_STRIDE 41984
#define SMEM_SMEM_INV_OFF 29696
#define SMEM_SMEM_INV_STAGE_BYTES 2048
#define SMEM_SMEM_INV_STRIDE 41984
#define SMEM_SMEM_V_OFF 32384
#define SMEM_SMEM_V_STAGE_BYTES 8192
#define SMEM_SMEM_V_STRIDE 41984
#define SMEM_SMEM_KI_OFF 17408
#define SMEM_SMEM_KI_STAGE_BYTES 8192
#define SMEM_SMEM_KI_STRIDE 41984
#define SMEM_SMEM_GATE_OFF 25600
#define SMEM_SMEM_GATE_STAGE_BYTES 16384
#define SMEM_SMEM_GATE_STRIDE 41984
#define SMEM_SMEM_BETA_RAW_OFF 41984
#define SMEM_SMEM_BETA_RAW_STAGE_BYTES 512
#define SMEM_SMEM_BETA_RAW_STRIDE 41984
#define SMEM_SMEM_INV_WORK_OFF 32384
#define SMEM_SMEM_INV_WORK_STAGE_BYTES 4096
#define SMEM_SMEM_INV_WORK_STRIDE 41984
#define SMEM_SMEM_OUT_OFF 210944
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_RESTORE_FACTOR_ALL_OFF 41984
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES 168452
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STRIDE 168452
#define SMEM_SMEM_GT_PREFIX_ALL_OFF 41472
#define SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES 168448
#define SMEM_SMEM_GT_PREFIX_ALL_STRIDE 168448
#define SMEM_SMEM_GT_ALL_OFF 31744
#define SMEM_SMEM_GT_ALL_STAGE_BYTES 168448
#define SMEM_SMEM_GT_ALL_STRIDE 168448
#define SMEM_SMEM_PREP_BETA_ALL_OFF 42500
#define SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES 168064
#define SMEM_SMEM_PREP_BETA_ALL_STRIDE 168064
#define SMEM_SMEM_GATE_RATE_ALL_OFF 42628
#define SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES 167940
#define SMEM_SMEM_GATE_RATE_ALL_STRIDE 167940
#define SMEM_SMEM_V_ALL_OFF 32384
#define SMEM_SMEM_V_ALL_STAGE_BYTES 176128
#define SMEM_SMEM_V_ALL_STRIDE 176128
#define SMEM_SMEM_GATE_ALL_OFF 25600
#define SMEM_SMEM_GATE_ALL_STAGE_BYTES 184320
#define SMEM_SMEM_GATE_ALL_STRIDE 184320
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_OFF 227328
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STAGE_BYTES 80
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STRIDE 80
#define SMEM_TOTAL 227456
#define THREADS 1024
#define STORE_BACKWARD_TAPE 1
#define STORE_E_TAPE 0

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_flashkda_bf16_fused_m128(__nv_bfloat16* __restrict__ q, FlashKDATensorMap const* q_tma, __nv_bfloat16* __restrict__ k, FlashKDATensorMap const* k_tma, __nv_bfloat16* __restrict__ v, FlashKDATensorMap const* v_tma, __nv_bfloat16* __restrict__ g, FlashKDATensorMap const* g_tma, __nv_bfloat16* __restrict__ beta, FlashKDATensorMap const* beta_tma, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ out, FlashKDATensorMap const* out_tma, __nv_bfloat16* __restrict__ final_state, int num_heads, int use_initial_state, int store_final_state, float scale, float lower_bound, unsigned long long state_indices_addr, unsigned long long state_checkpoints_addr, unsigned long long checkpoint_cu_starts_addr, long long beta_token_stride, long long state_slot_stride, int use_state_indices, int checkpoint_every_n_tokens, long long* __restrict__ cu_chunk_offsets, __nv_bfloat16* __restrict__ chunk_state, unsigned int* __restrict__ state_checkpoint_needed, __nv_bfloat16* __restrict__ tape_qd, __nv_bfloat16* __restrict__ tape_kd, __nv_bfloat16* __restrict__ tape_kr, __nv_bfloat16* __restrict__ tape_j, float* __restrict__ tape_restore_factor, __nv_bfloat16* __restrict__ tape_e, __nv_bfloat16* __restrict__ tape_x, __nv_bfloat16* __restrict__ tape_r, float* __restrict__ norm_inv_out, __nv_bfloat16* __restrict__ decay_out, float* __restrict__ beta_active_out, float* __restrict__ initial_state_f32, unsigned int* __restrict__ zero_workspace, int zero_words)
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
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(q_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(k_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(v_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(g_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(beta_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(out_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_qd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_all_addr = smem + 1024;
    __nv_bfloat16* smem_kd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kd_addr = smem + 9216;
    __nv_bfloat16* smem_q_raw_prefetch = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_q_raw_prefetch_addr = smem + 17408;
    __nv_bfloat16* smem_final_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_final_trans_addr = smem + 17408;
    __nv_bfloat16* smem_kr_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_kr_trans_addr = smem + 17408;
    __nv_bfloat16* smem_mqk_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int smem_mqk_trans_addr = smem + 25600;
    __nv_bfloat16* smem_inv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 29696);
    const int smem_inv_addr = smem + 29696;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_v_addr = smem + 32384;
    __nv_bfloat16* smem_ki = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_ki_addr = smem + 17408;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_gate_addr = smem + 25600;
    __nv_bfloat16* smem_beta_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_beta_raw_addr = smem + 41984;
    __nv_bfloat16* smem_inv_work = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_inv_work_addr = smem + 32384;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 210944);
    const int smem_out_addr = smem + 210944;
    float* smem_restore_factor_all = reinterpret_cast<float*>(smem_raw + 41984);
    const int smem_restore_factor_all_addr = smem + 41984;
    float* smem_gt_prefix_all = reinterpret_cast<float*>(smem_raw + 41472);
    const int smem_gt_prefix_all_addr = smem + 41472;
    float* smem_gt_all = reinterpret_cast<float*>(smem_raw + 31744);
    const int smem_gt_all_addr = smem + 31744;
    float* smem_prep_beta_all = reinterpret_cast<float*>(smem_raw + 42500);
    const int smem_prep_beta_all_addr = smem + 42500;
    float* smem_gate_rate_all = reinterpret_cast<float*>(smem_raw + 42628);
    const int smem_gate_rate_all_addr = smem + 42628;
    __nv_bfloat16* smem_v_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_v_all_addr = smem + 32384;
    float* smem_gate_all = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_gate_all_addr = smem + 25600;
    unsigned int* smem_state_checkpoint_needed = reinterpret_cast<unsigned int*>(smem_raw + 227328);
    const int smem_state_checkpoint_needed_addr = smem + 227328;

    // Mbarrier init (19 groups, 87 barriers)
    // Mbarriers at smem_raw[0..696)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'chunk_pipe' ---
            // qk_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // tape_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // tape_free: 5 barriers, init_count=2
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            // gate_raw_full: 5 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // qk_raw_full: 5 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // v_full: 5 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // v_free: 5 barriers, init_count=4
            mbarrier_init(smem + 240, 4);
            mbarrier_init(smem + 248, 4);
            mbarrier_init(smem + 256, 4);
            mbarrier_init(smem + 264, 4);
            mbarrier_init(smem + 272, 4);
            // smem_free: 5 barriers, init_count=1
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // raw_inputs_free: 5 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            // state_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 360, 4);
            mbarrier_init(smem + 368, 4);
            mbarrier_init(smem + 376, 4);
            mbarrier_init(smem + 384, 4);
            mbarrier_init(smem + 392, 4);
            // old_out_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 400, 1);
            mbarrier_init(smem + 408, 1);
            mbarrier_init(smem + 416, 1);
            mbarrier_init(smem + 424, 1);
            mbarrier_init(smem + 432, 1);
            // u_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 440, 4);
            mbarrier_init(smem + 448, 4);
            mbarrier_init(smem + 456, 4);
            mbarrier_init(smem + 464, 4);
            mbarrier_init(smem + 472, 4);
            // u2_acc_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 480, 1);
            mbarrier_init(smem + 488, 1);
            mbarrier_init(smem + 496, 1);
            mbarrier_init(smem + 504, 1);
            mbarrier_init(smem + 512, 1);
            // u2_inp_ready: 5 barriers, init_count=4
            mbarrier_init(smem + 520, 4);
            mbarrier_init(smem + 528, 4);
            mbarrier_init(smem + 536, 4);
            mbarrier_init(smem + 544, 4);
            mbarrier_init(smem + 552, 4);
            // final_ready: 5 barriers, init_count=1
            mbarrier_init(smem + 560, 1);
            mbarrier_init(smem + 568, 1);
            mbarrier_init(smem + 576, 1);
            mbarrier_init(smem + 584, 1);
            mbarrier_init(smem + 592, 1);
            // out_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 600, 1);
            // tmem_dealloc_ready: 1 barriers, init_count=2
            mbarrier_init(smem + 608, 2);
            // prep_diag_ready: 5 barriers, init_count=2
            mbarrier_init(smem + 616, 2);
            mbarrier_init(smem + 624, 2);
            mbarrier_init(smem + 632, 2);
            mbarrier_init(smem + 640, 2);
            mbarrier_init(smem + 648, 2);
            // prep_inv16_ready: 5 barriers, init_count=2
            mbarrier_init(smem + 656, 2);
            mbarrier_init(smem + 664, 2);
            mbarrier_init(smem + 672, 2);
            mbarrier_init(smem + 680, 2);
            mbarrier_init(smem + 688, 2);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 696);
    if (warp == 0) {
        int _tmem_hold = smem + 696;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define qk_full_addr (mbar_base + 0)
    #define tape_ready_addr (mbar_base + 40)
    #define tape_free_addr (mbar_base + 80)
    #define gate_raw_full_addr (mbar_base + 120)
    #define qk_raw_full_addr (mbar_base + 160)
    #define v_full_addr (mbar_base + 200)
    #define v_free_addr (mbar_base + 240)
    #define smem_free_addr (mbar_base + 280)
    #define raw_inputs_free_addr (mbar_base + 320)
    #define state_inp_ready_addr (mbar_base + 360)
    #define old_out_ready_addr (mbar_base + 400)
    #define u_inp_ready_addr (mbar_base + 440)
    #define u2_acc_ready_addr (mbar_base + 480)
    #define u2_inp_ready_addr (mbar_base + 520)
    #define final_ready_addr (mbar_base + 560)
    #define out_empty_addr (mbar_base + 600)
    #define tmem_dealloc_ready_addr (mbar_base + 608)
    #define prep_diag_ready_addr (mbar_base + 616)
    #define prep_inv16_ready_addr (mbar_base + 656)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr + 64;
    const int tmem_tmem_state_inp = taddr;
    const int tmem_tmem_u_acc = taddr + 224;
    const int tmem_tmem_u2_inp = taddr + 224;
    const int tmem_tmem_u2_acc = taddr;
    const int tmem_tmem_out = taddr + 192;
    const int tmem_tmem_state_out = taddr + 64;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: compute ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // compute_main
            int task_idx = blockIdx.x;
            int seq_idx = seq_order[task_idx / num_heads];
            int head_idx = task_idx % num_heads;
            long long bos = cu_seqlens[seq_idx];
            long long eos = cu_seqlens[seq_idx + 1];
            int seq_len = (int)(eos - bos);
            int num_chunks = (seq_len + 32 - 1) / 32;
            int num_sequences = num_bids / num_heads;
            long long total_chunks = cu_chunk_offsets[num_sequences];
            long long fallback_head = total_chunks * (long long)num_heads + (long long)seq_idx * (long long)num_heads + (long long)head_idx;
            int warp_in_wg = warp % 4;
            const int tmem_row_base = warp_in_wg * 32 << 16;
            int state_row = warp_in_wg * 32 + lane;
            int warp_id_in_role = (warp - 0);
            int compute_local_warp = warp_id_in_role;
            long long state_base = (((long long)seq_idx * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
            #pragma unroll
            for (int state_col_block = 0; state_col_block < 4; state_col_block++) {
                float state_frag[32];
                state_frag[0] = 0.0f;
                state_frag[1] = 0.0f;
                state_frag[2] = 0.0f;
                state_frag[3] = 0.0f;
                state_frag[4] = 0.0f;
                state_frag[5] = 0.0f;
                state_frag[6] = 0.0f;
                state_frag[7] = 0.0f;
                state_frag[8] = 0.0f;
                state_frag[9] = 0.0f;
                state_frag[10] = 0.0f;
                state_frag[11] = 0.0f;
                state_frag[12] = 0.0f;
                state_frag[13] = 0.0f;
                state_frag[14] = 0.0f;
                state_frag[15] = 0.0f;
                state_frag[16] = 0.0f;
                state_frag[17] = 0.0f;
                state_frag[18] = 0.0f;
                state_frag[19] = 0.0f;
                state_frag[20] = 0.0f;
                state_frag[21] = 0.0f;
                state_frag[22] = 0.0f;
                state_frag[23] = 0.0f;
                state_frag[24] = 0.0f;
                state_frag[25] = 0.0f;
                state_frag[26] = 0.0f;
                state_frag[27] = 0.0f;
                state_frag[28] = 0.0f;
                state_frag[29] = 0.0f;
                state_frag[30] = 0.0f;
                state_frag[31] = 0.0f;
                if (use_initial_state != 0) {
                    {
                        float initial_values[8];
                        #pragma unroll
                        for (int initial_quarter = 0; initial_quarter < 4; initial_quarter++) {
                            {
                                unsigned _ldv8_0_0;
                                unsigned _ldv8_0_1;
                                unsigned _ldv8_0_2;
                                unsigned _ldv8_0_3;
                                unsigned _ldv8_0_4;
                                unsigned _ldv8_0_5;
                                unsigned _ldv8_0_6;
                                unsigned _ldv8_0_7;
                                asm volatile(
                                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(initial_state_f32 + (state_base + (long long)(state_col_block * 32) + (long long)(initial_quarter * 8)))) : "memory");
                                initial_values[0 + 0] = __uint_as_float(_ldv8_0_0);
                                initial_values[0 + 1] = __uint_as_float(_ldv8_0_1);
                                initial_values[0 + 2] = __uint_as_float(_ldv8_0_2);
                                initial_values[0 + 3] = __uint_as_float(_ldv8_0_3);
                                initial_values[0 + 4] = __uint_as_float(_ldv8_0_4);
                                initial_values[0 + 5] = __uint_as_float(_ldv8_0_5);
                                initial_values[0 + 6] = __uint_as_float(_ldv8_0_6);
                                initial_values[0 + 7] = __uint_as_float(_ldv8_0_7);
                            }
                            #pragma unroll
                            for (int initial_item = 0; initial_item < 8; initial_item++) {
                                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(initial_values[initial_item]);
                                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_0);
                                state_frag[initial_quarter * 8 + initial_item] = _cvt_f32_2;
                            }
                        }
                    }
                }
                tmem_st_x32_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block * 32), state_frag);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            {
                if (compute_local_warp == 0) {
                    if (elect_sync()) {
                        long long first_chunk_head = cu_chunk_offsets[seq_idx] * (long long)num_heads + (long long)head_idx;
                        state_checkpoint_needed[first_chunk_head] = 0;
                        state_checkpoint_needed[fallback_head] = 0;
                    }
                }
            }
            unsigned int compute_stage = 0;
            unsigned int _phase_qk_full = 0;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_old_out_ready = 0;
            unsigned int _phase_u2_acc_ready = 0;
            unsigned int _phase_final_ready = 0;
            #pragma unroll 1
            for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                mbarrier_wait(qk_full_addr + (compute_stage) * 8, _phase_qk_full);
                #pragma unroll 1
                for (int state_col_block_1 = 0; state_col_block_1 < 4; state_col_block_1++) {
                    int state_addr = taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_1 * 32);
                    float _tmem_load_1[32];
                    tmem_ld_x32(&_tmem_load_1[0], state_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    uint32_t _tmem_load_1_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_1 * 16)), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[15]))
                        : "memory");
                    float state_scale[16];
                    #pragma unroll
                    for (int state_half = 0; state_half < 2; state_half++) {
                        #pragma unroll
                        for (int state_col = 0; state_col < 16; state_col++) {
                            state_scale[state_col] = smem_gt_all[compute_stage * 10496 + (unsigned int)(state_col_block_1 * 32) + (unsigned int)(state_half * 16) + (unsigned int)state_col];
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + state_half * 16))[_ls], reinterpret_cast<const float2*>(state_scale)[_ls]);
                    }
                    tmem_st_x32_f32(state_addr, _tmem_load_1);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(state_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(v_full_addr + (compute_stage) * 8, _phase_v_full);
                mbarrier_wait(old_out_ready_addr + (compute_stage) * 8, _phase_old_out_ready);
                float _tmem_load_2[32];
                tmem_ld_x32(&_tmem_load_2[0], taddr + 224 + (unsigned int)tmem_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                long long chunk_global_e = cu_chunk_offsets[seq_idx] + (long long)chunk_idx;
                long long tape_ex_base = ((chunk_global_e * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 32;
                #pragma unroll
                for (int residual_half = 0; residual_half < 2; residual_half++) {
                    float residual_v[16];
                    float residual_beta[16];
                    #pragma unroll
                    for (int residual_col = 0; residual_col < 16; residual_col++) {
                        int token_col = residual_half * 16 + residual_col;
                        __nv_bfloat16 v_value = smem_v_all[compute_stage * 20992 + (unsigned int)(token_col * 128) + (unsigned int)state_row];
                        float _cvt_f32_3 = __bfloat162float(v_value);
                        residual_v[residual_col] = _cvt_f32_3;
                        residual_beta[residual_col] = smem_prep_beta_all[compute_stage * 10496 + (unsigned int)token_col];
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        sub_f32x2_inplace(&reinterpret_cast<float2*>(residual_v)[_ls], reinterpret_cast<const float2*>((_tmem_load_2 + residual_half * 16))[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(residual_v)[_ls], reinterpret_cast<const float2*>(residual_beta)[_ls]);
                    {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(residual_v[0 + 0], residual_v[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(residual_v[0 + 2], residual_v[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(residual_v[0 + 4], residual_v[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(residual_v[0 + 6], residual_v[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_x + (tape_ex_base + (long long)(residual_half * 16))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(residual_v[8 + 0], residual_v[8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(residual_v[8 + 2], residual_v[8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(residual_v[8 + 4], residual_v[8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(residual_v[8 + 6], residual_v[8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_x + (tape_ex_base + (long long)(residual_half * 16) + 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    uint32_t residual_v_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 0], residual_v[_lp*2+1 + 0]));
                        residual_v_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base + (unsigned int)(residual_half * 8), (const uint32_t*)residual_v_bf16);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(v_free_addr + (compute_stage) * 8);
                    mbarrier_arrive(u_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(u2_acc_ready_addr + (compute_stage) * 8, _phase_u2_acc_ready);
                float _tmem_load_3[32];
                tmem_ld_x32(&_tmem_load_3[0], taddr + (unsigned int)tmem_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                {
                    long long chunk_global_r = cu_chunk_offsets[seq_idx] + (long long)chunk_idx;
                    long long tape_r_base = ((chunk_global_r * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 32;
                    #pragma unroll
                    for (int tape_r_vec = 0; tape_r_vec < 4; tape_r_vec++) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 0], _tmem_load_3[tape_r_vec * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 2], _tmem_load_3[tape_r_vec * 8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 4], _tmem_load_3[tape_r_vec * 8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 6], _tmem_load_3[tape_r_vec * 8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_r + (tape_r_base + (long long)(tape_r_vec * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
                uint32_t _tmem_load_3_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + 224 + (unsigned int)tmem_row_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3_bf16[15]))
                    : "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(u2_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(final_ready_addr + (compute_stage) * 8, _phase_final_ready);
                {
                    if (num_chunks > chunk_idx + 1) {
                        int checkpoint_needed = smem_state_checkpoint_needed[compute_stage * 4] != 0 || smem_state_checkpoint_needed[compute_stage * 4 + 1] != 0 || smem_state_checkpoint_needed[compute_stage * 4 + 2] != 0 || smem_state_checkpoint_needed[compute_stage * 4 + 3] != 0;
                        if (compute_local_warp == 0) {
                            if (elect_sync()) {
                                long long chunk_global_next = cu_chunk_offsets[seq_idx] + (long long)chunk_idx + 1;
                                state_checkpoint_needed[chunk_global_next * (long long)num_heads + (long long)head_idx] = (unsigned int)checkpoint_needed;
                                if (checkpoint_needed != 0) {
                                    state_checkpoint_needed[fallback_head] = 1;
                                }
                            }
                        }
                    }
                    if (compute_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(smem_free_addr + (compute_stage) * 8);
                        }
                    }
                }
                compute_stage += 1;
                if (compute_stage == 5) { compute_stage = 0; _phase_qk_full ^= 1; _phase_v_full ^= 1; _phase_old_out_ready ^= 1; _phase_u2_acc_ready ^= 1; _phase_final_ready ^= 1; }
            }
            if (store_final_state != 0) {
                #pragma unroll
                for (int state_col_block_2 = 0; state_col_block_2 < 4; state_col_block_2++) {
                    float _tmem_load_5[32];
                    tmem_ld_x32(&_tmem_load_5[0], taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_5[0 + 0], _tmem_load_5[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_5[0 + 2], _tmem_load_5[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_5[0 + 4], _tmem_load_5[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_5[0 + 6], _tmem_load_5[0 + 7]);
                        _pk[4] = __floats2bfloat162_rn(_tmem_load_5[0 + 8], _tmem_load_5[0 + 9]);
                        _pk[5] = __floats2bfloat162_rn(_tmem_load_5[0 + 10], _tmem_load_5[0 + 11]);
                        _pk[6] = __floats2bfloat162_rn(_tmem_load_5[0 + 12], _tmem_load_5[0 + 13]);
                        _pk[7] = __floats2bfloat162_rn(_tmem_load_5[0 + 14], _tmem_load_5[0 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base + (long long)(state_col_block_2 * 32))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base + (long long)(state_col_block_2 * 32))))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                    {
                        __nv_bfloat162 _pk[8];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_5[16 + 0], _tmem_load_5[16 + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_5[16 + 2], _tmem_load_5[16 + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_5[16 + 4], _tmem_load_5[16 + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_5[16 + 6], _tmem_load_5[16 + 7]);
                        _pk[4] = __floats2bfloat162_rn(_tmem_load_5[16 + 8], _tmem_load_5[16 + 9]);
                        _pk[5] = __floats2bfloat162_rn(_tmem_load_5[16 + 10], _tmem_load_5[16 + 11]);
                        _pk[6] = __floats2bfloat162_rn(_tmem_load_5[16 + 12], _tmem_load_5[16 + 13]);
                        _pk[7] = __floats2bfloat162_rn(_tmem_load_5[16 + 14], _tmem_load_5[16 + 15]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base + (long long)(state_col_block_2 * 32) + 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(final_state + (state_base + (long long)(state_col_block_2 * 32) + 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                    }
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (compute_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // epilogue_main
            int task_idx_1 = blockIdx.x;
            int seq_idx_1 = seq_order[task_idx_1 / num_heads];
            int head_idx_1 = task_idx_1 % num_heads;
            long long bos_1 = cu_seqlens[seq_idx_1];
            long long eos_1 = cu_seqlens[seq_idx_1 + 1];
            int seq_len_1 = (int)(eos_1 - bos_1);
            int num_chunks_1 = (seq_len_1 + 32 - 1) / 32;
            int warp_id_in_role_1 = (warp - 4);
            int epilogue_local_warp = warp_id_in_role_1;
            int warp_in_wg_1 = warp % 4;
            const int tmem_row_base_1 = warp_in_wg_1 * 32 << 16;
            int state_row_1 = warp_in_wg_1 * 32 + lane;
            unsigned int epilogue_stage = 0;
            unsigned int output_stage = 0;
            int epilogue_chunks = num_chunks_1;
            {
                epilogue_chunks = 0;
            }
            unsigned int _phase_final_ready_1 = 0;
            #pragma unroll 1
            for (int chunk_idx_1 = 0; chunk_idx_1 < epilogue_chunks; chunk_idx_1++) {
                int chunk_is_full = ((seq_len_1 >= (chunk_idx_1 + 1) * 32) ? 1 : 0);
                if (chunk_is_full != 0) {
                    mbarrier_wait(final_ready_addr + (epilogue_stage) * 8, _phase_final_ready_1);
                    float _tmem_load_6[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1)
                        : "memory");
                    float _tmem_load_7[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1 + 1048576)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    if (epilogue_local_warp == 0) {
                        if (chunk_idx_1 >= 2) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    int out_stage_addr = smem_out_addr + output_stage * 8192;
                    #pragma unroll
                    for (int dim_half = 0; dim_half < 2; dim_half++) {
                        unsigned int out_packed[8];
                        if (dim_half == 0) {
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        } else {
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                        #pragma unroll
                        for (int token_group = 0; token_group < 2; token_group++) {
                            int mtx_idx = lane / 8;
                            int row_addr = lane & 7;
                            int dim_base = epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8;
                            int token_base = token_group * 16 + mtx_idx / 2 * 8;
                            int token_addr = token_base + row_addr;
                            int token_pair = token_addr / 2;
                            int token_parity = token_addr & 1;
                            int raw_row = token_pair + dim_base / 64 * 16;
                            int raw_col = (dim_base & 63 ^ (token_pair & 3) << 4 ^ token_parity << 3) + token_parity * 64;
                            int stsm_offset = (raw_row * 128 + raw_col) * 2;
                            const int pack_base = token_group * 4;
                            uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(out_stage_addr + stsm_offset));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 1])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 2])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 3]))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            tma_store_4d(out_tma, 0, (int)(bos_1 + (long long)(chunk_idx_1 * 32)), head_idx_1, 0, smem_out_addr + output_stage * 8192);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    output_stage = output_stage ^ 1;
                } else {
                    mbarrier_wait(final_ready_addr + (epilogue_stage) * 8, _phase_final_ready_1);
                    float _tmem_load_8[32];
                    tmem_ld_x32(&_tmem_load_8[0], taddr + 192 + (unsigned int)tmem_row_base_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    #pragma unroll
                    for (int token_col_1 = 0; token_col_1 < 32; token_col_1++) {
                        long long out_token = bos_1 + (long long)(chunk_idx_1 * 32 + token_col_1);
                        if (out_token < eos_1) {
                            long long out_idx = (out_token * (long long)num_heads + (long long)head_idx_1) * 128 + (long long)state_row_1;
                            out[out_idx] = _tmem_load_8[token_col_1];
                        }
                    }
                }
                epilogue_stage += 1;
                if (epilogue_stage == 5) { epilogue_stage = 0; _phase_final_ready_1 ^= 1; }
            }
            if (epilogue_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int task_idx_2 = blockIdx.x;
            int seq_idx_2 = seq_order[task_idx_2 / num_heads];
            long long bos_2 = cu_seqlens[seq_idx_2];
            long long eos_2 = cu_seqlens[seq_idx_2 + 1];
            int seq_len_2 = (int)(eos_2 - bos_2);
            int num_chunks_2 = (seq_len_2 + 32 - 1) / 32;
            unsigned int mma_stage = 0;
            unsigned int _phase_qk_full_1 = 0;
            unsigned int _phase_state_inp_ready = 0;
            unsigned int _phase_out_empty_0 = 1;
            unsigned int _phase_u_inp_ready = 0;
            unsigned int _phase_u2_inp_ready = 0;
            unsigned int _phase_final_ready_2 = 0;
            #pragma unroll 1
            for (int _chunk_idx = 0; _chunk_idx < num_chunks_2; _chunk_idx++) {
                mbarrier_wait(qk_full_addr + (mma_stage) * 8, _phase_qk_full_1);
                mbarrier_wait(state_inp_ready_addr + (mma_stage) * 8, _phase_state_inp_ready);
                int _mma_b_lo_1 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + (mma_stage) * 2624);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u_acc), "r"(_mma_b_lo_1), "r"(tmem_tmem_state_inp), "r"(0));
                elect_commit2(old_out_ready_addr + (mma_stage) * 8, raw_inputs_free_addr + (mma_stage) * 8);
                mbarrier_wait(u_inp_ready_addr + (mma_stage) * 8, _phase_u_inp_ready);
                int _mma_b_lo_2 = make_warp_uniform((((smem_inv_addr) >> 4) & 0x3FFF) + (mma_stage) * 2624);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u2_acc), "r"(_mma_b_lo_2), "r"(tmem_tmem_u2_inp), "r"(0));
                elect_commit(u2_acc_ready_addr + (mma_stage) * 8);
                mbarrier_wait(u2_inp_ready_addr + (mma_stage) * 8, _phase_u2_inp_ready);
                int _mma_b_lo_3 = make_warp_uniform(((((smem_final_trans_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_stage) * 2624);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136905872;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_state_out), "r"(_mma_b_lo_3), "r"(tmem_tmem_u2_inp), "r"(1));
                elect_commit(final_ready_addr + (mma_stage) * 8);
                {
                    mbarrier_wait(final_ready_addr + (mma_stage) * 8, _phase_final_ready_2);
                }
                mma_stage += 1;
                if (mma_stage == 5) { mma_stage = 0; _phase_qk_full_1 ^= 1; _phase_state_inp_ready ^= 1; _phase_u_inp_ready ^= 1; _phase_u2_inp_ready ^= 1; _phase_final_ready_2 ^= 1; }
            }
            unsigned int _phase_tmem_dealloc_ready_0 = 0;
            mbarrier_wait(tmem_dealloc_ready_addr, _phase_tmem_dealloc_ready_0);
            _phase_tmem_dealloc_ready_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }
    // ---- Role: tape ----
    if (warp >= 10 && warp <= 11) {
        { // tape_main
            unsigned int _phase_tape_ready = 0;
            {
                int task_idx_3 = blockIdx.x;
                int seq_idx_3 = seq_order[task_idx_3 / num_heads];
                int head_idx_2 = task_idx_3 % num_heads;
                long long bos_3 = cu_seqlens[seq_idx_3];
                long long eos_3 = cu_seqlens[seq_idx_3 + 1];
                int seq_len_3 = (int)(eos_3 - bos_3);
                int num_chunks_3 = (seq_len_3 + 32 - 1) / 32;
                int warp_id_in_role_2 = (warp - 10);
                int tape_tid = warp_id_in_role_2 * 32 + lane;
                int zero_stride = num_bids * 64;
                #pragma unroll 1
                for (int zero_item = task_idx_3 * 64 + tape_tid; zero_item < zero_words; zero_item += zero_stride) {
                    zero_workspace[zero_item] = 0;
                }
                unsigned int tape_stage = 0;
                #pragma unroll 1
                for (int chunk_idx_2 = 0; chunk_idx_2 < num_chunks_3; chunk_idx_2++) {
                    mbarrier_wait(tape_ready_addr + (tape_stage) * 8, _phase_tape_ready);
                    long long chunk_global = cu_chunk_offsets[seq_idx_3] + (long long)chunk_idx_2;
                    long long tape_vec_base = (chunk_global * (long long)num_heads + (long long)head_idx_2) * 32 * 128;
                    #pragma unroll
                    for (int tape_pass = 0; tape_pass < 8; tape_pass++) {
                        int tape_item = tape_pass * 64 + tape_tid;
                        int tape_row = tape_item / 16;
                        int tape_segment = tape_item % 16;
                        float tape_kr_values[8];
                        unsigned int packed[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                            : "r"((smem_kr_trans_addr + tape_stage * 41984 + (unsigned int)(tape_segment * 8 / 64 * 4096 + tape_row * 128 + tape_segment * 8 % 64 * 2 ^ (tape_segment * 8 / 64 * 4096 + tape_row * 128 + tape_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_f32[_pair * 2])[0]), "=f"((&packed_f32[_pair * 2])[1])
                                : "r"(packed[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx = 0; value_idx < 8; value_idx++) {
                            tape_kr_values[value_idx] = packed_f32[value_idx];
                        }
                        long long tape_index = tape_vec_base + (long long)tape_row * 128 + (long long)(tape_segment * 8);
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(tape_kr_values[0 + 0], tape_kr_values[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(tape_kr_values[0 + 2], tape_kr_values[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(tape_kr_values[0 + 4], tape_kr_values[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(tape_kr_values[0 + 6], tape_kr_values[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_kr + tape_index))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    long long tape_j_base = (chunk_global * (long long)num_heads + (long long)head_idx_2) * 32 * 32;
                    #pragma unroll
                    for (int tape_j_pass = 0; tape_j_pass < 2; tape_j_pass++) {
                        int tape_j_item = tape_j_pass * 64 + tape_tid;
                        int tape_j_row = tape_j_item / 4;
                        int tape_j_segment = tape_j_item % 4;
                        float tape_j_values[8];
                        unsigned int packed_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 3]))
                            : "r"((smem_inv_addr + tape_stage * 41984 + (unsigned int)(tape_j_segment * 8 / 16 * 1024 + tape_j_row * 32 + tape_j_segment * 8 % 16 * 2 ^ (tape_j_segment * 8 / 16 * 1024 + tape_j_row * 32 + tape_j_segment * 8 % 16 * 2 >> 7 & 1) << 4))));
                        float packed_f32_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_f32_1[_pair * 2])[0]), "=f"((&packed_f32_1[_pair * 2])[1])
                                : "r"(packed_1[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_1 = 0; value_idx_1 < 8; value_idx_1++) {
                            tape_j_values[value_idx_1] = packed_f32_1[value_idx_1];
                        }
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(tape_j_values[0 + 0], tape_j_values[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(tape_j_values[0 + 2], tape_j_values[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(tape_j_values[0 + 4], tape_j_values[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(tape_j_values[0 + 6], tape_j_values[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_j + (tape_j_base + (long long)tape_j_row * 32 + (long long)(tape_j_segment * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(tape_free_addr + (tape_stage) * 8);
                    }
                    tape_stage += 1;
                    if (tape_stage == 5) { tape_stage = 0; _phase_tape_ready ^= 1; }
                }
            }
        }
    }
    // ---- Role: prep ----
    if (warp >= 12 && warp <= 31) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // prep_main
            int task_idx_4 = blockIdx.x;
            int seq_idx_4 = seq_order[task_idx_4 / num_heads];
            int head_idx_3 = task_idx_4 % num_heads;
            long long bos_4 = cu_seqlens[seq_idx_4];
            long long eos_4 = cu_seqlens[seq_idx_4 + 1];
            int seq_len_4 = (int)(eos_4 - bos_4);
            int num_chunks_4 = (seq_len_4 + 32 - 1) / 32;
            int instance_id = (warp - 12) / 4;
            int prep_instance = instance_id;
            int warp_id_in_role_3 = (warp - 12);
            int prep_local_warp = warp_id_in_role_3 - prep_instance * 4;
            int prep_tid = prep_local_warp * 32 + lane;
            int num_prep_iters = (num_chunks_4 + 4 - prep_instance) / 5;
            unsigned int prep_stage = (unsigned int)prep_instance;
            int gate_rate_stage_f32 = prep_instance * 10496;
            if (prep_tid == 0) {
                float _expf_0 = __expf(A_log[head_idx_3]);
                smem_gate_rate_all[gate_rate_stage_f32] = _expf_0;
            }
            asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
            unsigned int _phase_raw_inputs_free = 1;
            unsigned int _phase_gate_raw_full = 0;
            unsigned int _phase_smem_free = 1;
            unsigned int _phase_v_free = 1;
            unsigned int _phase_qk_raw_full = 0;
            unsigned int _phase_prep_diag_ready = 0;
            unsigned int _phase_prep_inv16_ready = 0;
            unsigned int _phase_tape_free = 1;
            #pragma unroll 1
            for (int prep_iter = 0; prep_iter < num_prep_iters; prep_iter++) {
                int chunk_idx_3 = prep_iter * 5 + prep_instance;
                int stage_f32 = prep_stage * 10496;
                int stage_bf16 = prep_stage * 20992;
                int chunk_is_full_1 = ((seq_len_4 >= (chunk_idx_3 + 1) * 32) ? 1 : 0);
                float early_beta_value = 0.0f;
                float early_gate0 = 0.0f;
                if (chunk_is_full_1 != 0 || prep_iter != 0) {
                    mbarrier_wait(raw_inputs_free_addr + (prep_stage) * 8, _phase_raw_inputs_free);
                }
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(gate_raw_full_addr + (prep_stage) * 8, 8704);
                            tma_3d_gmem2smem(smem_g_raw_addr + prep_stage * 41984, g_tma, 0, head_idx_3, (int)(bos_4 + (long long)(chunk_idx_3 * 32)), gate_raw_full_addr + (prep_stage) * 8);
                            tma_2d_gmem2smem(smem_beta_raw_addr + prep_stage * 41984, beta_tma, head_idx_3 / 8 * 8, (int)(bos_4 + (long long)(chunk_idx_3 * 32)), gate_raw_full_addr + (prep_stage) * 8);
                            mbarrier_arrive_expect_tx(qk_raw_full_addr + (prep_stage) * 8, 16384);
                            tma_4d_gmem2smem(smem_kd_addr + prep_stage * 41984, k_tma, 0, (int)(bos_4 + (long long)(chunk_idx_3 * 32)), head_idx_3, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                    mbarrier_wait(gate_raw_full_addr + (prep_stage) * 8, _phase_gate_raw_full);
                    if (prep_local_warp == 2 && lane < 32) {
                        unsigned int beta_raw_pair[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&beta_raw_pair[0])) : "r"(smem_beta_raw_addr + prep_stage * 41984 + (unsigned int)(lane * 16) + (unsigned int)(head_idx_3 % 8 / 2 * 4)));
                        float beta_raw_pair_f32[2];
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&beta_raw_pair_f32[_pair * 2])[0]), "=f"((&beta_raw_pair_f32[_pair * 2])[1])
                                : "r"(beta_raw_pair[_pair]));
                        }
                        float beta_logit = beta_raw_pair_f32[0];
                        if (head_idx_3 % 2 != 0) {
                            beta_logit = beta_raw_pair_f32[1];
                        }
                        float _tanh_approx_0;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(beta_logit * 0.5f));
                        early_beta_value = _tanh_approx_0 * 0.5f + 0.5f;
                    }
                    if (prep_tid < 128) {
                        float early_gate_rate = smem_gate_rate_all[stage_f32];
                        float early_gate_bias = dt_bias[head_idx_3 * 128 + prep_tid];
                        __nv_bfloat16 early_gate_raw = smem_g_raw_all[stage_bf16 + prep_tid];
                        float _cvt_f32_0 = __bfloat162float(early_gate_raw);
                        float early_gate_arg = early_gate_rate * (_cvt_f32_0 + early_gate_bias);
                        float _tanh_approx_1;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(early_gate_arg * 0.5f));
                        float early_gate_sigmoid = _tanh_approx_1 * 0.5f + 0.5f;
                        early_gate0 = lower_bound * 1.4426950408889634f * early_gate_sigmoid;
                    }
                }
                mbarrier_wait(smem_free_addr + (prep_stage) * 8, _phase_smem_free);
                mbarrier_wait(v_free_addr + (prep_stage) * 8, _phase_v_free);
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            tma_4d_gmem2smem(smem_q_raw_prefetch_addr + prep_stage * 41984, q_tma, 0, (int)(bos_4 + (long long)(chunk_idx_3 * 32)), head_idx_3, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                }
                if (chunk_is_full_1 == 0) {
                    #pragma unroll
                    for (int gate_load_pass = 0; gate_load_pass < 4; gate_load_pass++) {
                        int gate_load_item = gate_load_pass * 128 + prep_tid;
                        int gate_load_row = gate_load_item / 16;
                        int gate_load_segment = gate_load_item % 16;
                        long long gate_load_token = bos_4 + (long long)(chunk_idx_3 * 32 + gate_load_row);
                        long long gate_load_base = (gate_load_token * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)(gate_load_segment * 8);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(smem_g_raw_addr + prep_stage * 41984 + (unsigned int)(gate_load_item * 16)), "l"(g + gate_load_base), "r"((gate_load_token < eos_4) ? 16 : 0));
                    }
                }
                if (chunk_is_full_1 == 0) {
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                }
                if (prep_local_warp == 2 && lane < 32) {
                    long long beta_token = bos_4 + (long long)(chunk_idx_3 * 32 + lane);
                    float beta_value = early_beta_value;
                    if (chunk_is_full_1 == 0) {
                        if (beta_token < eos_4) {
                            float beta_logit_1 = (float)beta[beta_token * (long long)num_heads + (long long)head_idx_3];
                            float _tanh_approx_2;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(beta_logit_1 * 0.5f));
                            beta_value = _tanh_approx_2 * 0.5f + 0.5f;
                        }
                    }
                    smem_prep_beta_all[stage_f32 + lane] = beta_value;
                    {
                        if (beta_token < eos_4) {
                            beta_active_out[beta_token * (long long)num_heads + (long long)head_idx_3] = beta_value;
                        }
                    }
                }
                if (prep_tid < 128) {
                    int gate_col = prep_tid;
                    float gate_rate = smem_gate_rate_all[stage_f32];
                    float gate_bias = dt_bias[head_idx_3 * 128 + gate_col];
                    float prefix_log2 = 0.0f;
                    for (int gate_row = 0; gate_row < 32; gate_row++) {
                        long long gate_token = bos_4 + (long long)(chunk_idx_3 * 32 + gate_row);
                        float gate_log2 = 0.0f;
                        int gate_needs_compute = 1;
                        if (gate_row == 0) {
                            if (chunk_is_full_1 != 0) {
                                gate_log2 = early_gate0;
                                gate_needs_compute = 0;
                            }
                        }
                        if (gate_needs_compute != 0) {
                            if (gate_token < eos_4) {
                                __nv_bfloat16 gate_raw = smem_g_raw_all[stage_bf16 + gate_row * 128 + gate_col];
                                float _cvt_f32_1 = __bfloat162float(gate_raw);
                                float gate_arg = gate_rate * (_cvt_f32_1 + gate_bias);
                                float _tanh_approx_3;
                                asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(gate_arg * 0.5f));
                                float gate_sigmoid = _tanh_approx_3 * 0.5f + 0.5f;
                                gate_log2 = lower_bound * 1.4426950408889634f * gate_sigmoid;
                            }
                        }
                        prefix_log2 += gate_log2;
                        smem_gate_all[stage_f32 + gate_row * 128 + gate_col] = prefix_log2;
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                if (chunk_is_full_1 != 0) {
                    mbarrier_wait(qk_raw_full_addr + (prep_stage) * 8, _phase_qk_raw_full);
                }
                if (prep_tid < 128) {
                    float total_log2 = smem_gt_prefix_all[stage_f32 + prep_tid];
                    float _exp2_0 = approx_exp2(total_log2 - lower_bound * 1.4426950408889634f * 16.0f);
                    float restore_factor_value = _exp2_0;
                    smem_restore_factor_all[stage_f32 + prep_tid] = restore_factor_value;
                    {
                        long long chunk_global_restore = cu_chunk_offsets[seq_idx_4] + (long long)chunk_idx_3;
                        tape_restore_factor[(chunk_global_restore * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)prep_tid] = restore_factor_value;
                        int _vote_0 = __any_sync(0xFFFFFFFF, num_chunks_4 > chunk_idx_3 + 1 && total_log2 > -30.0f);
                        int warp_slow = _vote_0;
                        if (lane == 0) {
                            smem_state_checkpoint_needed[prep_stage * 4 + (unsigned int)prep_local_warp] = (unsigned int)warp_slow;
                        }
                    }
                }
                if (prep_tid == 0) {
                    float _exp2_1 = approx_exp2(lower_bound * 1.4426950408889634f * 16.0f);
                    smem_restore_factor_all[stage_f32 + 128] = _exp2_1;
                }
                #pragma unroll 1
                for (int work_pass = 0; work_pass < 4; work_pass++) {
                    int work_item = work_pass * 128 + prep_tid;
                    int row = work_item / 16;
                    int segment = work_item % 16;
                    long long token = bos_4 + (long long)(chunk_idx_3 * 32 + row);
                    int token_valid = ((token < eos_4) ? 1 : 0);
                    long long gmem_base = (token * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)(segment * 8);
                    float q_raw_vec[8];
                    float k_raw_vec[8];
                    q_raw_vec[0] = 0.0f;
                    q_raw_vec[1] = 0.0f;
                    q_raw_vec[2] = 0.0f;
                    q_raw_vec[3] = 0.0f;
                    q_raw_vec[4] = 0.0f;
                    q_raw_vec[5] = 0.0f;
                    q_raw_vec[6] = 0.0f;
                    q_raw_vec[7] = 0.0f;
                    k_raw_vec[0] = 0.0f;
                    k_raw_vec[1] = 0.0f;
                    k_raw_vec[2] = 0.0f;
                    k_raw_vec[3] = 0.0f;
                    k_raw_vec[4] = 0.0f;
                    k_raw_vec[5] = 0.0f;
                    k_raw_vec[6] = 0.0f;
                    k_raw_vec[7] = 0.0f;
                    if (chunk_is_full_1 != 0) {
                        unsigned int packed_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 3]))
                            : "r"((smem_q_raw_prefetch_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_f32_2[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_f32_2[_pair * 2])[0]), "=f"((&packed_f32_2[_pair * 2])[1])
                                : "r"(packed_2[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_2 = 0; value_idx_2 < 8; value_idx_2++) {
                            q_raw_vec[value_idx_2] = packed_f32_2[value_idx_2];
                        }
                        unsigned int packed_0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 3]))
                            : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                        float packed_0_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_0_f32[_pair * 2])[0]), "=f"((&packed_0_f32[_pair * 2])[1])
                                : "r"(packed_0[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_3 = 0; value_idx_3 < 8; value_idx_3++) {
                            k_raw_vec[value_idx_3] = packed_0_f32[value_idx_3];
                        }
                    } else if (token_valid != 0) {
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q + gmem_base);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&q_raw_vec[0 + _blk * 8 + _pair * 2])[0]), "=f"((&q_raw_vec[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(k + gmem_base);
                            uint4 _vld_1[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_1[_blk] = _vptr_1[_blk];
                                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&k_raw_vec[0 + _blk * 8 + _pair * 2])[0]), "=f"((&k_raw_vec[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_1[_pair]));
                                }
                            }
                        }
                    }
                    float q_sum = 0.0f;
                    float k_sum = 0.0f;
                    for (int elem_in_segment = 0; elem_in_segment < 8; elem_in_segment++) {
                        float q_raw = q_raw_vec[elem_in_segment];
                        float k_raw = k_raw_vec[elem_in_segment];
                        float _fma_0 = __fmaf_rn(q_raw, q_raw, q_sum);
                        q_sum = _fma_0;
                        float _fma_1 = __fmaf_rn(k_raw, k_raw, k_sum);
                        k_sum = _fma_1;
                    }
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 8);
                    q_sum += _shfl_xor_0;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 8);
                    k_sum += _shfl_xor_1;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 4);
                    q_sum += _shfl_xor_2;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 4);
                    k_sum += _shfl_xor_3;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 2);
                    q_sum += _shfl_xor_4;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 2);
                    k_sum += _shfl_xor_5;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 1);
                    q_sum += _shfl_xor_6;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 1);
                    k_sum += _shfl_xor_7;
                    float _rsqrt_0 = rsqrtf(q_sum + 1e-06f);
                    float q_inv = _rsqrt_0;
                    float _rsqrt_1 = rsqrtf(k_sum + 1e-06f);
                    float k_inv = _rsqrt_1;
                    const float2 _scale2_2 = {q_inv, q_inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(q_raw_vec)[_ls], _scale2_2);
                    const float2 _scale2_3 = {k_inv, k_inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(k_raw_vec)[_ls], _scale2_3);
                    {
                        if (token_valid != 0) {
                            if (segment == 0) {
                                float norm_pair[2];
                                norm_pair[0] = q_inv;
                                norm_pair[1] = k_inv;
                                {
                                    float2 _v2 = make_float2(norm_pair[0 + 0], norm_pair[0 + 1]);
                                    *reinterpret_cast<float2*>(norm_inv_out + (token * (long long)num_heads + (long long)head_idx_3) * 2) = _v2;
                                }
                            }
                        }
                    }
                    float qd_vec[8];
                    float kd_vec[8];
                    float ki_vec[8];
                    for (int elem_in_segment_1 = 0; elem_in_segment_1 < 8; elem_in_segment_1++) {
                        int col = segment * 8 + elem_in_segment_1;
                        float prefix = smem_gate_all[stage_f32 + row * 128 + col];
                        float common_log2 = lower_bound * 1.4426950408889634f * 16.0f;
                        float _exp2_2 = approx_exp2(prefix - common_log2);
                        float decay = _exp2_2;
                        qd_vec[elem_in_segment_1] = decay;
                        kd_vec[elem_in_segment_1] = decay;
                        ki_vec[elem_in_segment_1] = k_raw_vec[elem_in_segment_1] / decay;
                    }
                    {
                        if (token_valid != 0) {
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(qd_vec[0 + 0], qd_vec[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(qd_vec[0 + 2], qd_vec[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(qd_vec[0 + 4], qd_vec[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(qd_vec[0 + 6], qd_vec[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(decay_out))[gmem_base + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], reinterpret_cast<const float2*>(q_raw_vec)[_ls]);
                    const float2 _scale2_4 = {scale, scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], _scale2_4);
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(kd_vec)[_ls], reinterpret_cast<const float2*>(k_raw_vec)[_ls]);
                    unsigned int packed_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                        packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word * 4)), "r"((packed_3[word])));
                    }
                    unsigned int packed_0_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_vec[_lp*2 + 0], kd_vec[_lp*2+1 + 0]));
                        packed_0_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_1 = 0; word_1 < 4; word_1++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_1 * 4)), "r"((packed_0_1[word_1])));
                    }
                    unsigned int packed_1_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_vec[_lp*2 + 0], ki_vec[_lp*2+1 + 0]));
                        packed_1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_2 = 0; word_2 < 4; word_2++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_2 * 4)), "r"((packed_1_1[word_2])));
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                unsigned int a_frag[4];
                unsigned int b_frag[4];
                float acc[8];
                {
                    int pair_row_base = prep_local_warp / 2 * 16;
                    int pair_col_base = prep_local_warp % 2 * 16;
                    if (pair_row_base >= pair_col_base) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0 = pair_row_base + lane / 4;
                        int row1 = row0 + 8;
                        int col0 = pair_col_base + lane % 4 * 2;
                        float beta0 = smem_prep_beta_all[stage_f32 + row0];
                        float beta1 = smem_prep_beta_all[stage_f32 + row1];
                        float seed[8];
                        seed[0] = 0.0f;
                        seed[1] = 0.0f;
                        seed[2] = 0.0f;
                        seed[3] = 0.0f;
                        seed[4] = 0.0f;
                        seed[5] = 0.0f;
                        seed[6] = 0.0f;
                        seed[7] = 0.0f;
                        if (row0 > col0) {
                            seed[0] = acc[0] * beta0;
                        }
                        if (row0 > col0 + 1) {
                            seed[1] = acc[1] * beta0;
                        }
                        if (row1 > col0) {
                            seed[2] = acc[2] * beta1;
                        }
                        if (row1 > col0 + 1) {
                            seed[3] = acc[3] * beta1;
                        }
                        if (row0 > col0 + 8) {
                            seed[4] = acc[4] * beta0;
                        }
                        if (row0 > col0 + 9) {
                            seed[5] = acc[5] * beta0;
                        }
                        if (row1 > col0 + 8) {
                            seed[6] = acc[6] * beta1;
                        }
                        if (row1 > col0 + 9) {
                            seed[7] = acc[7] * beta1;
                        }
                        unsigned int seed_packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(seed[_lp*2 + 0], seed[_lp*2+1 + 0]));
                            seed_packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        int seed_lane_row = lane % 16;
                        int seed_lane_col = lane / 16 * 8;
                        int byte_off = (int)prep_stage * 41984 + (pair_row_base + seed_lane_row) * 128 + (pair_col_base + seed_lane_col) * 2;
                        int swizzled_off = byte_off ^ (byte_off >> 7 & 7) << 4;
                        int seed_addr = smem_inv_work_addr + (unsigned int)swizzled_off;
                        uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)seed_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    } else {
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        acc[4] = 0.0f;
                        acc[5] = 0.0f;
                        acc[6] = 0.0f;
                        acc[7] = 0.0f;
                    }
                    int row0_1 = pair_row_base + lane / 4;
                    int row1_1 = row0_1 + 8;
                    int col0_1 = pair_col_base + lane % 4 * 2;
                    float mqk[8];
                    mqk[0] = 0.0f;
                    mqk[1] = 0.0f;
                    mqk[2] = 0.0f;
                    mqk[3] = 0.0f;
                    mqk[4] = 0.0f;
                    mqk[5] = 0.0f;
                    mqk[6] = 0.0f;
                    mqk[7] = 0.0f;
                    if (row0_1 >= col0_1) {
                        mqk[0] = acc[0];
                    }
                    if (row0_1 >= col0_1 + 1) {
                        mqk[1] = acc[1];
                    }
                    if (row1_1 >= col0_1) {
                        mqk[2] = acc[2];
                    }
                    if (row1_1 >= col0_1 + 1) {
                        mqk[3] = acc[3];
                    }
                    if (row0_1 >= col0_1 + 8) {
                        mqk[4] = acc[4];
                    }
                    if (row0_1 >= col0_1 + 9) {
                        mqk[5] = acc[5];
                    }
                    if (row1_1 >= col0_1 + 8) {
                        mqk[6] = acc[6];
                    }
                    if (row1_1 >= col0_1 + 9) {
                        mqk[7] = acc[7];
                    }
                    unsigned int mqk_packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mqk[_lp*2 + 0], mqk[_lp*2+1 + 0]));
                        mqk_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int publish_pair = 0; publish_pair < 2; publish_pair++) {
                        int publish_row = pair_col_base + publish_pair * 8 + (lane & 7);
                        int publish_col = 128 + pair_row_base + lane / 8 * 8;
                        uint32_t _stmatrix_addr_6 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + prep_stage * 41984 + (unsigned int)(publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 ^ (publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                            :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2 + 1]))
                            : "memory");
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                long long tape_scaled_base = 0;
                {
                    long long chunk_global_scaled = cu_chunk_offsets[seq_idx_4] + (long long)chunk_idx_3;
                    tape_scaled_base = (chunk_global_scaled * (long long)num_heads + (long long)head_idx_3) * 32 * 128;
                }
                if (prep_tid < 128) {
                    float total_log2_1 = smem_gt_prefix_all[stage_f32 + prep_tid];
                    float _exp2_3 = approx_exp2(total_log2_1);
                    smem_gt_all[stage_f32 + prep_tid] = _exp2_3;
                }
                {
                    if (prep_local_warp >= 2) {
                        int stage_f32_0 = prep_stage * 10496;
                        float restore_scale = smem_restore_factor_all[stage_f32_0 + 128];
                        float restore_factor[8];
                        int restore_segment = lane & 15;
                        #pragma unroll
                        for (int restore_elem = 0; restore_elem < 8; restore_elem++) {
                            int restore_col = restore_segment * 8 + restore_elem;
                            restore_factor[restore_elem] = smem_restore_factor_all[stage_f32_0 + restore_col];
                        }
                        #pragma unroll 1
                        for (int restore_pass = 0; restore_pass < 6; restore_pass++) {
                            int restore_row = 8 + (prep_local_warp - 2) * 12 + restore_pass * 2 + (lane >> 4);
                            float restore_qd_values[8];
                            float restore_kd_values[8];
                            float restore_ki_values[8];
                            unsigned int packed_4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 3]))
                                : "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_f32_3[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_f32_3[_pair * 2])[0]), "=f"((&packed_f32_3[_pair * 2])[1])
                                    : "r"(packed_4[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_4 = 0; value_idx_4 < 8; value_idx_4++) {
                                restore_qd_values[value_idx_4] = packed_f32_3[value_idx_4];
                            }
                            unsigned int packed_0_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 3]))
                                : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_0_f32_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_0_f32_1[_pair * 2])[0]), "=f"((&packed_0_f32_1[_pair * 2])[1])
                                    : "r"(packed_0_2[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_5 = 0; value_idx_5 < 8; value_idx_5++) {
                                restore_kd_values[value_idx_5] = packed_0_f32_1[value_idx_5];
                            }
                            unsigned int packed_1_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 3]))
                                : "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_1_f32[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_1_f32[_pair * 2])[0]), "=f"((&packed_1_f32[_pair * 2])[1])
                                    : "r"(packed_1_2[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_6 = 0; value_idx_6 < 8; value_idx_6++) {
                                restore_ki_values[value_idx_6] = packed_1_f32[value_idx_6];
                            }
                            {
                                long long tape_scaled_index = tape_scaled_base + (long long)restore_row * 128 + (long long)(restore_segment * 8);
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(restore_qd_values[0 + 0], restore_qd_values[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(restore_qd_values[0 + 2], restore_qd_values[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(restore_qd_values[0 + 4], restore_qd_values[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(restore_qd_values[0 + 6], restore_qd_values[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_qd + tape_scaled_index))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(restore_kd_values[0 + 0], restore_kd_values[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(restore_kd_values[0 + 2], restore_kd_values[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(restore_kd_values[0 + 4], restore_kd_values[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(restore_kd_values[0 + 6], restore_kd_values[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_kd + tape_scaled_index))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                            float restore_kr_values[8];
                            #pragma unroll
                            for (int restore_elem_1 = 0; restore_elem_1 < 8; restore_elem_1++) {
                                restore_kr_values[restore_elem_1] = restore_ki_values[restore_elem_1] * restore_factor[restore_elem_1];
                            }
                            const float2 _scale2_7 = {restore_scale, restore_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_qd_values)[_ls], _scale2_7);
                            const float2 _scale2_8 = {restore_scale, restore_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_kd_values)[_ls], _scale2_8);
                            unsigned int packed_2_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values[_lp*2 + 0], restore_kd_values[_lp*2+1 + 0]));
                                packed_2_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_3 = 0; word_3 < 4; word_3++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_3 * 4)), "r"((packed_2_1[word_3])));
                            }
                            unsigned int packed_3_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values[_lp*2 + 0], restore_kr_values[_lp*2+1 + 0]));
                                packed_3_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_4 = 0; word_4 < 4; word_4++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_4 * 4)), "r"((packed_3_1[word_4])));
                            }
                        }
                    } else if (prep_local_warp == 1) {
                        int stage_f32_0_1 = prep_stage * 10496;
                        float restore_scale_1 = smem_restore_factor_all[stage_f32_0_1 + 128];
                        float restore_factor_1[8];
                        int restore_segment_1 = lane & 15;
                        #pragma unroll
                        for (int restore_elem_2 = 0; restore_elem_2 < 8; restore_elem_2++) {
                            int restore_col_1 = restore_segment_1 * 8 + restore_elem_2;
                            restore_factor_1[restore_elem_2] = smem_restore_factor_all[stage_f32_0_1 + restore_col_1];
                        }
                        #pragma unroll 1
                        for (int restore_pass_1 = 0; restore_pass_1 < 4; restore_pass_1++) {
                            int restore_row_1 = restore_pass_1 * 2 + (lane >> 4);
                            float restore_qd_values_1[8];
                            float restore_kd_values_1[8];
                            float restore_ki_values_1[8];
                            unsigned int packed_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 3]))
                                : "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_f32_4[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_f32_4[_pair * 2])[0]), "=f"((&packed_f32_4[_pair * 2])[1])
                                    : "r"(packed_5[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_7 = 0; value_idx_7 < 8; value_idx_7++) {
                                restore_qd_values_1[value_idx_7] = packed_f32_4[value_idx_7];
                            }
                            unsigned int packed_0_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 3]))
                                : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_0_f32_2[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_0_f32_2[_pair * 2])[0]), "=f"((&packed_0_f32_2[_pair * 2])[1])
                                    : "r"(packed_0_3[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_8 = 0; value_idx_8 < 8; value_idx_8++) {
                                restore_kd_values_1[value_idx_8] = packed_0_f32_2[value_idx_8];
                            }
                            unsigned int packed_1_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 3]))
                                : "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_1_f32_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_1_f32_1[_pair * 2])[0]), "=f"((&packed_1_f32_1[_pair * 2])[1])
                                    : "r"(packed_1_3[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_9 = 0; value_idx_9 < 8; value_idx_9++) {
                                restore_ki_values_1[value_idx_9] = packed_1_f32_1[value_idx_9];
                            }
                            {
                                long long tape_scaled_index_1 = tape_scaled_base + (long long)restore_row_1 * 128 + (long long)(restore_segment_1 * 8);
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(restore_qd_values_1[0 + 0], restore_qd_values_1[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(restore_qd_values_1[0 + 2], restore_qd_values_1[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(restore_qd_values_1[0 + 4], restore_qd_values_1[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(restore_qd_values_1[0 + 6], restore_qd_values_1[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_qd + tape_scaled_index_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(restore_kd_values_1[0 + 0], restore_kd_values_1[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(restore_kd_values_1[0 + 2], restore_kd_values_1[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(restore_kd_values_1[0 + 4], restore_kd_values_1[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(restore_kd_values_1[0 + 6], restore_kd_values_1[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_kd + tape_scaled_index_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                            float restore_kr_values_1[8];
                            #pragma unroll
                            for (int restore_elem_3 = 0; restore_elem_3 < 8; restore_elem_3++) {
                                restore_kr_values_1[restore_elem_3] = restore_ki_values_1[restore_elem_3] * restore_factor_1[restore_elem_3];
                            }
                            const float2 _scale2_9 = {restore_scale_1, restore_scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_qd_values_1)[_ls], _scale2_9);
                            const float2 _scale2_10 = {restore_scale_1, restore_scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(restore_kd_values_1)[_ls], _scale2_10);
                            unsigned int packed_2_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values_1[_lp*2 + 0], restore_kd_values_1[_lp*2+1 + 0]));
                                packed_2_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_5 = 0; word_5 < 4; word_5++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_5 * 4)), "r"((packed_2_2[word_5])));
                            }
                            unsigned int packed_3_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values_1[_lp*2 + 0], restore_kr_values_1[_lp*2+1 + 0]));
                                packed_3_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_6 = 0; word_6 < 4; word_6++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_6 * 4)), "r"((packed_3_2[word_6])));
                            }
                        }
                    }
                }
                if (prep_local_warp == 0) {
                    int inverse_row = lane;
                    int diag_block = inverse_row / 8;
                    int lane_in_diag = lane & 7;
                    float inv_row[8];
                    {
                        unsigned int packed_6[4];
                        int byte_off_1 = (int)prep_stage * 41984 + inverse_row * 128 + diag_block * 8 * 2;
                        int swizzled_off_1 = byte_off_1 ^ (byte_off_1 >> 7 & 7) << 4;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 3]))
                            : "r"(smem_inv_work_addr + (unsigned int)swizzled_off_1));
                        float packed_f32_5[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&packed_f32_5[_pair * 2])[0]), "=f"((&packed_f32_5[_pair * 2])[1])
                                : "r"(packed_6[_pair]));
                        }
                        #pragma unroll
                        for (int value_idx_10 = 0; value_idx_10 < 8; value_idx_10++) {
                            inv_row[value_idx_10] = packed_f32_5[value_idx_10];
                        }
                    }
                    #pragma unroll
                    for (int diag_elem = 0; diag_elem < 8; diag_elem++) {
                        if (lane_in_diag == diag_elem) {
                            inv_row[diag_elem] = 1.0f;
                        }
                    }
                    int diag_group_base = lane - lane_in_diag;
                    #pragma unroll
                    for (int src_row = 0; src_row < 7; src_row++) {
                        float row_scale = -inv_row[src_row];
                        #pragma unroll
                        for (int prev_col = 0; prev_col < src_row; prev_col++) {
                            int pivot_lane = diag_group_base + src_row;
                            float _shfl_0 = __shfl_sync(0xFFFFFFFF, inv_row[prev_col], pivot_lane);
                            float pivot = _shfl_0;
                            if (lane_in_diag > src_row) {
                                float _fma_2 = __fmaf_rn(row_scale, pivot, inv_row[prev_col]);
                                inv_row[prev_col] = _fma_2;
                            }
                        }
                        if (lane_in_diag > src_row) {
                            inv_row[src_row] = row_scale;
                        }
                    }
                    {
                        unsigned int packed_7[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv_row[_lp*2 + 0], inv_row[_lp*2+1 + 0]));
                            packed_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        int byte_off_2 = (int)prep_stage * 41984 + inverse_row * 128 + diag_block * 8 * 2;
                        int swizzled_off_2 = byte_off_2 ^ (byte_off_2 >> 7 & 7) << 4;
                        #pragma unroll
                        for (int word_7 = 0; word_7 < 4; word_7++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_inv_work_addr + (unsigned int)swizzled_off_2 + (unsigned int)(word_7 * 4)), "r"((packed_7[word_7])));
                        }
                    }
                }
                if (prep_local_warp < 2) {
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(prep_diag_ready_addr + (prep_stage) * 8);
                    }
                    mbarrier_wait(prep_diag_ready_addr + (prep_stage) * 8, _phase_prep_diag_ready);
                }
                {
                    if (prep_local_warp < 2) {
                        int lane_row = lane & 7;
                        int byte_off_3 = (int)prep_stage * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + (prep_local_warp * 16 + 8) * 2;
                        int swizzled_off_3 = byte_off_3 ^ (byte_off_3 >> 7 & 7) << 4;
                        int d_addr = smem_inv_work_addr + (unsigned int)swizzled_off_3;
                        int byte_off_0 = (int)prep_stage * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                        int swizzled_off_1_1 = byte_off_0 ^ (byte_off_0 >> 7 & 7) << 4;
                        int c_addr = smem_inv_work_addr + (unsigned int)swizzled_off_1_1;
                        int byte_off_2_1 = (int)prep_stage * 41984 + (prep_local_warp * 16 + lane_row) * 128 + prep_local_warp * 16 * 2;
                        int swizzled_off_3_1 = byte_off_2_1 ^ (byte_off_2_1 >> 7 & 7) << 4;
                        int a_addr = smem_inv_work_addr + (unsigned int)swizzled_off_3_1;
                        unsigned int d_frag[2];
                        unsigned int c_frag[1];
                        float dc_acc[4];
                        unsigned int dc_bf16[2];
                        unsigned int inv_a_frag[1];
                        float o_acc[4];
                        unsigned int o_bf16[2];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                            : "=r"(d_frag[0])
                            : "r"(d_addr)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                            : "=r"(d_frag[1])
                            : "r"(d_addr)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                            : "=r"(c_frag[0])
                            : "r"(c_addr)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(dc_acc[0]), "=f"(dc_acc[1]), "=f"(dc_acc[2]), "=f"(dc_acc[3])
                            : "r"(d_frag[0]), "r"(d_frag[1]), "r"(c_frag[0]));
                        const float2 _scale2_11 = {-1.0f, -1.0f};
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(dc_acc)[_ls], _scale2_11);
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dc_acc[_lp*2 + 0], dc_acc[_lp*2+1 + 0]));
                            dc_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                            : "=r"(inv_a_frag[0])
                            : "r"(a_addr)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(o_acc[0]), "=f"(o_acc[1]), "=f"(o_acc[2]), "=f"(o_acc[3])
                            : "r"(dc_bf16[0]), "r"(dc_bf16[1]), "r"(inv_a_frag[0]));
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_acc[_lp*2 + 0], o_acc[_lp*2+1 + 0]));
                            o_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int byte_off_4 = (int)prep_stage * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                        int swizzled_off_5 = byte_off_4 ^ (byte_off_4 >> 7 & 7) << 4;
                        int o_addr = smem_inv_work_addr + (unsigned int)swizzled_off_5;
                        uint32_t _stmatrix_addr_12 = static_cast<uint32_t>((unsigned long long)o_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                            :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&o_bf16[0]))
                            : "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(prep_inv16_ready_addr + (prep_stage) * 8);
                        }
                        mbarrier_wait(prep_inv16_ready_addr + (prep_stage) * 8, _phase_prep_inv16_ready);
                    }
                }
                {
                    if (prep_local_warp == 0) {
                        int lane_row_1 = lane % 16;
                        int lane_col = lane / 16 * 8;
                        int byte_off_5 = (int)prep_stage * 41984 + (16 + lane_row_1) * 128 + (16 + lane_col) * 2;
                        int swizzled_off_4 = byte_off_5 ^ (byte_off_5 >> 7 & 7) << 4;
                        int d_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_4;
                        int byte_off_0_1 = (int)prep_stage * 41984 + (16 + lane_row_1) * 128 + lane_col * 2;
                        int swizzled_off_1_2 = byte_off_0_1 ^ (byte_off_0_1 >> 7 & 7) << 4;
                        int c_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_1_2;
                        int byte_off_2_2 = (int)prep_stage * 41984 + lane_row_1 * 128 + lane_col * 2;
                        int swizzled_off_3_2 = byte_off_2_2 ^ (byte_off_2_2 >> 7 & 7) << 4;
                        int a_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_3_2;
                        unsigned int d32_frag[4];
                        unsigned int c32_frag[4];
                        float dc32_acc[8];
                        unsigned int dc32_bf16[4];
                        unsigned int a32_frag[4];
                        float o32_acc[8];
                        unsigned int o32_bf16[4];
                        unsigned int zero32_bf16[4];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(d32_frag[0]), "=r"(d32_frag[1]), "=r"(d32_frag[2]), "=r"(d32_frag[3])
                            : "r"(d_addr_1)
                            : "memory");
                        int d_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_13 = static_cast<uint32_t>((unsigned long long)d_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(c32_frag[0]), "=r"(c32_frag[1]), "=r"(c32_frag[2]), "=r"(c32_frag[3])
                            : "r"(c_addr_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(dc32_acc[0]), "=f"(dc32_acc[1]), "=f"(dc32_acc[2]), "=f"(dc32_acc[3])
                            : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[0]), "r"(c32_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(dc32_acc[4]), "=f"(dc32_acc[(4) + 1]), "=f"(dc32_acc[(4) + 2]), "=f"(dc32_acc[(4) + 3])
                            : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[2]), "r"(c32_frag[(2) + 1]));
                        const float2 _scale2_14 = {-1.0f, -1.0f};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(dc32_acc)[_ls], _scale2_14);
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dc32_acc[_lp*2 + 0], dc32_acc[_lp*2+1 + 0]));
                            dc32_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a32_frag[0]), "=r"(a32_frag[1]), "=r"(a32_frag[2]), "=r"(a32_frag[3])
                            : "r"(a_addr_1)
                            : "memory");
                        int a_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)(lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_15 = static_cast<uint32_t>((unsigned long long)a_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[3]))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(o32_acc[0]), "=f"(o32_acc[1]), "=f"(o32_acc[2]), "=f"(o32_acc[3])
                            : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]), "r"(a32_frag[0]), "r"(a32_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(o32_acc[4]), "=f"(o32_acc[(4) + 1]), "=f"(o32_acc[(4) + 2]), "=f"(o32_acc[(4) + 3])
                            : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]), "r"(a32_frag[2]), "r"(a32_frag[(2) + 1]));
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o32_acc[_lp*2 + 0], o32_acc[_lp*2+1 + 0]));
                            o32_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int o_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)(lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_16 = static_cast<uint32_t>((unsigned long long)o_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[3]))
                            : "memory");
                        #pragma unroll
                        for (int zero_word = 0; zero_word < 4; zero_word++) {
                            zero32_bf16[zero_word] = 0;
                        }
                        int zero_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                        uint32_t _stmatrix_addr_17 = static_cast<uint32_t>((unsigned long long)zero_publish_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[3]))
                            : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                if (prep_local_warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(qk_full_addr + (prep_stage) * 8);
                        {
                            mbarrier_arrive(tape_ready_addr + (prep_stage) * 8);
                        }
                    }
                }
                {
                    mbarrier_wait(tape_free_addr + (prep_stage) * 8, _phase_tape_free);
                }
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(v_full_addr + (prep_stage) * 8, 8192);
                            tma_3d_gmem2smem(smem_v_addr + prep_stage * 41984, v_tma, 0, head_idx_3, (int)(bos_4 + (long long)(chunk_idx_3 * 32)), v_full_addr + (prep_stage) * 8);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int v_load_iter = 0; v_load_iter < 4; v_load_iter++) {
                        int v_item = v_load_iter * 128 + prep_tid;
                        int row_1 = v_item / 16;
                        int segment_1 = v_item % 16;
                        long long token_1 = bos_4 + (long long)(chunk_idx_3 * 32 + row_1);
                        int token_valid_1 = ((token_1 < eos_4) ? 1 : 0);
                        long long v_src = (token_1 * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)(segment_1 * 8);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(smem_v_addr + prep_stage * 41984 + (unsigned int)((row_1 * 128 + segment_1 * 8) * 2)), "l"(v + v_src), "r"((token_valid_1 != 0) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(v_full_addr + (prep_stage) * 8);
                        }
                    }
                }
                for (int _advance = 0; _advance < 5; _advance++) {
                    prep_stage += 1;
                    if (prep_stage == 5) { prep_stage = 0; _phase_raw_inputs_free ^= 1; _phase_gate_raw_full ^= 1; _phase_smem_free ^= 1; _phase_v_free ^= 1; _phase_qk_raw_full ^= 1; _phase_prep_diag_ready ^= 1; _phase_prep_inv16_ready ^= 1; _phase_tape_free ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef FLASHKDA_INF
#undef NUM_CHUNK_PIPE_STAGES
#undef SMEM_SMEM_BETA_RAW_OFF
#undef SMEM_SMEM_BETA_RAW_STAGE_BYTES
#undef SMEM_SMEM_BETA_RAW_STRIDE
#undef SMEM_SMEM_FINAL_TRANS_OFF
#undef SMEM_SMEM_FINAL_TRANS_STAGE_BYTES
#undef SMEM_SMEM_FINAL_TRANS_STRIDE
#undef SMEM_SMEM_GATE_ALL_OFF
#undef SMEM_SMEM_GATE_ALL_STAGE_BYTES
#undef SMEM_SMEM_GATE_ALL_STRIDE
#undef SMEM_SMEM_GATE_OFF
#undef SMEM_SMEM_GATE_RATE_ALL_OFF
#undef SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES
#undef SMEM_SMEM_GATE_RATE_ALL_STRIDE
#undef SMEM_SMEM_GATE_STAGE_BYTES
#undef SMEM_SMEM_GATE_STRIDE
#undef SMEM_SMEM_GT_ALL_OFF
#undef SMEM_SMEM_GT_ALL_STAGE_BYTES
#undef SMEM_SMEM_GT_ALL_STRIDE
#undef SMEM_SMEM_GT_PREFIX_ALL_OFF
#undef SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES
#undef SMEM_SMEM_GT_PREFIX_ALL_STRIDE
#undef SMEM_SMEM_G_RAW_ALL_OFF
#undef SMEM_SMEM_G_RAW_ALL_STAGE_BYTES
#undef SMEM_SMEM_G_RAW_ALL_STRIDE
#undef SMEM_SMEM_G_RAW_OFF
#undef SMEM_SMEM_G_RAW_STAGE_BYTES
#undef SMEM_SMEM_G_RAW_STRIDE
#undef SMEM_SMEM_INV_OFF
#undef SMEM_SMEM_INV_STAGE_BYTES
#undef SMEM_SMEM_INV_STRIDE
#undef SMEM_SMEM_INV_WORK_OFF
#undef SMEM_SMEM_INV_WORK_STAGE_BYTES
#undef SMEM_SMEM_INV_WORK_STRIDE
#undef SMEM_SMEM_KD_OFF
#undef SMEM_SMEM_KD_STAGE_BYTES
#undef SMEM_SMEM_KD_STRIDE
#undef SMEM_SMEM_KI_OFF
#undef SMEM_SMEM_KI_STAGE_BYTES
#undef SMEM_SMEM_KI_STRIDE
#undef SMEM_SMEM_KR_TRANS_OFF
#undef SMEM_SMEM_KR_TRANS_STAGE_BYTES
#undef SMEM_SMEM_KR_TRANS_STRIDE
#undef SMEM_SMEM_MQK_TRANS_OFF
#undef SMEM_SMEM_MQK_TRANS_STAGE_BYTES
#undef SMEM_SMEM_MQK_TRANS_STRIDE
#undef SMEM_SMEM_OUT_OFF
#undef SMEM_SMEM_OUT_STAGE_BYTES
#undef SMEM_SMEM_OUT_STRIDE
#undef SMEM_SMEM_PREP_BETA_ALL_OFF
#undef SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES
#undef SMEM_SMEM_PREP_BETA_ALL_STRIDE
#undef SMEM_SMEM_QD_OFF
#undef SMEM_SMEM_QD_STAGE_BYTES
#undef SMEM_SMEM_QD_STRIDE
#undef SMEM_SMEM_Q_RAW_PREFETCH_OFF
#undef SMEM_SMEM_Q_RAW_PREFETCH_STAGE_BYTES
#undef SMEM_SMEM_Q_RAW_PREFETCH_STRIDE
#undef SMEM_SMEM_RESTORE_FACTOR_ALL_OFF
#undef SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES
#undef SMEM_SMEM_RESTORE_FACTOR_ALL_STRIDE
#undef SMEM_SMEM_STATE_CHECKPOINT_NEEDED_OFF
#undef SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STAGE_BYTES
#undef SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STRIDE
#undef SMEM_SMEM_V_ALL_OFF
#undef SMEM_SMEM_V_ALL_STAGE_BYTES
#undef SMEM_SMEM_V_ALL_STRIDE
#undef SMEM_SMEM_V_OFF
#undef SMEM_SMEM_V_STAGE_BYTES
#undef SMEM_SMEM_V_STRIDE
#undef SMEM_TOTAL
#undef STORE_BACKWARD_TAPE
#undef STORE_E_TAPE
#undef THREADS
#undef TMEM_NCOLS
#undef TMEM_TMEM_OUT_OFFSET
#undef TMEM_TMEM_STATE_INP_OFFSET
#undef TMEM_TMEM_STATE_OFFSET
#undef TMEM_TMEM_STATE_OUT_OFFSET
#undef TMEM_TMEM_U2_ACC_OFFSET
#undef TMEM_TMEM_U2_INP_OFFSET
#undef TMEM_TMEM_U_ACC_OFFSET
#undef final_ready_addr
#undef gate_raw_full_addr
#undef old_out_ready_addr
#undef out_empty_addr
#undef prep_diag_ready_addr
#undef prep_inv16_ready_addr
#undef qk_full_addr
#undef qk_raw_full_addr
#undef raw_inputs_free_addr
#undef smem_beta_raw_addr
#undef smem_final_trans_addr
#undef smem_free_addr
#undef smem_g_raw_addr
#undef smem_g_raw_all_addr
#undef smem_gate_addr
#undef smem_gate_all_addr
#undef smem_gate_rate_all_addr
#undef smem_gt_all_addr
#undef smem_gt_prefix_all_addr
#undef smem_inv_addr
#undef smem_inv_work_addr
#undef smem_kd_addr
#undef smem_ki_addr
#undef smem_kr_trans_addr
#undef smem_mqk_trans_addr
#undef smem_out_addr
#undef smem_prep_beta_all_addr
#undef smem_q_raw_prefetch_addr
#undef smem_qd_addr
#undef smem_restore_factor_all_addr
#undef smem_state_checkpoint_needed_addr
#undef smem_v_addr
#undef smem_v_all_addr
#undef state_inp_ready_addr
#undef tape_free_addr
#undef tape_ready_addr
#undef tmem_dealloc_ready_addr
#undef u2_acc_ready_addr
#undef u2_inp_ready_addr
#undef u_inp_ready_addr
#undef v_free_addr
#undef v_full_addr

#define FLASHKDA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_ANY_FALLBACK_OFF 0
#define SMEM_ANY_FALLBACK_STAGE_BYTES 4
#define SMEM_ANY_FALLBACK_STRIDE 4
#define SMEM_TOTAL 128
#define THREADS 256
#define C 32
#define K 128
#define V 128

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashkda_backward_state_checkpoint_fallback_c32(long long* __restrict__ cu_seqlens, long long* __restrict__ cu_chunk_offsets, unsigned int* __restrict__ state_checkpoint_needed, float* __restrict__ initial_state, __nv_bfloat16* __restrict__ tape_kr, float* __restrict__ tape_restore_factor, __nv_bfloat16* __restrict__ tape_r, __nv_bfloat16* __restrict__ chunk_state, int num_sequences, int num_heads, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    unsigned int* any_fallback = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int any_fallback_addr = smem + 0;

    // === Task calls (dependency order) ===
    long long total_chunks = cu_chunk_offsets[num_sequences];
    int num_tasks = num_sequences * num_heads;
    if (tid == 0) {
        any_fallback[0] = 0;
    }
    asm volatile("barrier.sync 1, 256;" ::: "memory");
    #pragma unroll 1
    for (int scan_task = tid; scan_task < num_tasks; scan_task += 256) {
        if (state_checkpoint_needed[total_chunks * (long long)num_heads + (long long)scan_task] != 0) {
            atomicAdd(&any_fallback[0], 1);
        }
    }
    asm volatile("barrier.sync 2, 256;" ::: "memory");
    if (any_fallback[0] != 0) {
        #pragma unroll 1
        for (int task = 0; task < num_tasks; task++) {
            int sequence = task / num_heads;
            int head = task - sequence * num_heads;
            unsigned int head_fallback_needed = state_checkpoint_needed[total_chunks * (long long)num_heads + (long long)task];
            if (head_fallback_needed != 0) {
                long long bos = cu_seqlens[sequence];
                long long eos = cu_seqlens[sequence + 1];
                int num_chunks = ((int)(eos - bos) + C - 1) / C;
                int key_base = lane * 4;
                long long initial_base = ((long long)sequence * (long long)num_heads + (long long)head) * (long long)V * (long long)K;
                float _expf_0 = __expf(lower_bound * (float)(C / 2));
                float common_decay = _expf_0;
                int warp_id_in_role = (warp - 0);
                #pragma unroll 1
                for (int value_row = warp_id_in_role; value_row < V; value_row += 8) {
                    float state[4];
                    long long initial_row_base = initial_base + (long long)value_row * (long long)K + (long long)key_base;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(initial_state + initial_row_base);
                        state[0 + 0] = _v4.x;
                        state[0 + 1] = _v4.y;
                        state[0 + 2] = _v4.z;
                        state[0 + 3] = _v4.w;
                    }
                    #pragma unroll
                    for (int key_elem = 0; key_elem < 4; key_elem++) {
                        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(state[key_elem]);
                        float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                        state[key_elem] = _cvt_f32_0;
                    }
                    #pragma unroll 1
                    for (int local_chunk = 0; local_chunk < num_chunks; local_chunk++) {
                        long long chunk_global = cu_chunk_offsets[sequence] + (long long)local_chunk;
                        long long chunk_head = chunk_global * (long long)num_heads + (long long)head;
                        long long restore_base = chunk_head * (long long)K + (long long)key_base;
                        float restore_values[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(tape_restore_factor + restore_base);
                            restore_values[0 + 0] = _v4.x;
                            restore_values[0 + 1] = _v4.y;
                            restore_values[0 + 2] = _v4.z;
                            restore_values[0 + 3] = _v4.w;
                        }
                        #pragma unroll
                        for (int key_elem2 = 0; key_elem2 < 4; key_elem2++) {
                            state[key_elem2] = state[key_elem2] * (restore_values[key_elem2] * common_decay);
                        }
                        long long kr_base = chunk_head * (long long)C * (long long)K + (long long)key_base;
                        long long r_base = (chunk_head * (long long)V + (long long)value_row) * (long long)C;
                        #pragma unroll
                        for (int token_local = 0; token_local < C; token_local++) {
                            float kr_values[4];
                            {
                                uint2 _vld_2;
                                _vld_2 = *reinterpret_cast<const uint2*>(tape_kr + kr_base + (long long)(token_local * K));
                                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
                                #pragma unroll
                                for (int _pair = 0; _pair < 2; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&kr_values[0 + _pair * 2])[0]), "=f"((&kr_values[0 + _pair * 2])[1])
                                        : "r"(_vpairs_2[_pair]));
                                }
                            }
                            float r_value = (float)tape_r[r_base + (long long)token_local];
                            #pragma unroll
                            for (int key_elem3 = 0; key_elem3 < 4; key_elem3++) {
                                float _fma_0 = __fmaf_rn(r_value, kr_values[key_elem3], state[key_elem3]);
                                state[key_elem3] = _fma_0;
                            }
                        }
                        if (num_chunks > local_chunk + 1) {
                            long long next_chunk_global = chunk_global + 1;
                            long long next_chunk_head = next_chunk_global * (long long)num_heads + (long long)head;
                            if (state_checkpoint_needed[next_chunk_head] != 0) {
                                long long checkpoint_base = (next_chunk_head * (long long)V + (long long)value_row) * (long long)K + (long long)key_base;
                                {
                                    uint2 _pk2;
                                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                    _pk[0] = __floats2bfloat162_rn(state[0 + 0], state[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(state[0 + 2], state[0 + 3]);
                                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(chunk_state + checkpoint_base))[0]) = _pk2;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef C
#undef K
#undef FLASHKDA_INF
#undef NUM_MAIN_STAGES
#undef SMEM_ANY_FALLBACK_OFF
#undef SMEM_ANY_FALLBACK_STAGE_BYTES
#undef SMEM_ANY_FALLBACK_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef V
#undef any_fallback_addr

#define FLASHKDA_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_TMEM_DH_OFFSET 64
#define TMEM_TMEM_DH_INP_OFFSET 0
#define TMEM_TMEM_DO_INITIAL_OFFSET 224
#define TMEM_TMEM_DO_FINAL_OFFSET 0
#define TMEM_TMEM_DR_OFFSET 192
#define TMEM_TMEM_DR_INP_OFFSET 192
#define TMEM_TMEM_DX_OFFSET 224
#define TMEM_TMEM_DE_INP_OFFSET 224
#define NUM_REVERSE_PIPE_STAGES 1
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 8192
#define SMEM_SMEM_QD_STRIDE 8192
#define SMEM_SMEM_QD_TRANS_OFF 1024
#define SMEM_SMEM_QD_TRANS_STAGE_BYTES 8192
#define SMEM_SMEM_QD_TRANS_STRIDE 8192
#define SMEM_SMEM_KD_OFF 9216
#define SMEM_SMEM_KD_STAGE_BYTES 8192
#define SMEM_SMEM_KD_STRIDE 8192
#define SMEM_SMEM_KD_TRANS_OFF 9216
#define SMEM_SMEM_KD_TRANS_STAGE_BYTES 8192
#define SMEM_SMEM_KD_TRANS_STRIDE 8192
#define SMEM_SMEM_KR_OFF 17408
#define SMEM_SMEM_KR_STAGE_BYTES 8192
#define SMEM_SMEM_KR_STRIDE 8192
#define SMEM_SMEM_KI_OFF 25600
#define SMEM_SMEM_KI_STAGE_BYTES 8192
#define SMEM_SMEM_KI_STRIDE 8192
#define SMEM_SMEM_J_OFF 33792
#define SMEM_SMEM_J_STAGE_BYTES 2048
#define SMEM_SMEM_J_STRIDE 2048
#define SMEM_SMEM_J_TRANS_OFF 33792
#define SMEM_SMEM_J_TRANS_STAGE_BYTES 2048
#define SMEM_SMEM_J_TRANS_STRIDE 2048
#define SMEM_SMEM_N_OFF 35840
#define SMEM_SMEM_N_STAGE_BYTES 2048
#define SMEM_SMEM_N_STRIDE 2048
#define SMEM_TOTAL 37888
#define THREADS 288

extern "C" {

__global__ __launch_bounds__(288) void
kernel_flashkda_backward_boundary_c32_tcgen_m64(__nv_bfloat16* __restrict__ do_, float* __restrict__ dfinal_state, float* __restrict__ beta_active, long long* __restrict__ cu_seqlens, long long* __restrict__ cu_chunk_offsets, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ tape_qd, __nv_bfloat16* __restrict__ tape_kd, __nv_bfloat16* __restrict__ tape_kr, __nv_bfloat16* __restrict__ tape_j, float* __restrict__ tape_restore_factor, __nv_bfloat16* __restrict__ chunk_dh, __nv_bfloat16* __restrict__ chunk_dr, __nv_bfloat16* __restrict__ chunk_dx, float* __restrict__ dinitial_state, unsigned int* __restrict__ boundary_ready, int num_heads, int publish_ready, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_qd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_qd_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_trans_addr = smem + 1024;
    __nv_bfloat16* smem_kd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kd_addr = smem + 9216;
    __nv_bfloat16* smem_kd_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kd_trans_addr = smem + 9216;
    __nv_bfloat16* smem_kr = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_kr_addr = smem + 17408;
    __nv_bfloat16* smem_ki = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int smem_ki_addr = smem + 25600;
    __nv_bfloat16* smem_j = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_j_addr = smem + 33792;
    __nv_bfloat16* smem_j_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_j_trans_addr = smem + 33792;
    __nv_bfloat16* smem_n = reinterpret_cast<__nv_bfloat16*>(smem_raw + 35840);
    const int smem_n_addr = smem + 35840;

    // Mbarrier init (9 groups, 9 barriers)
    // Mbarriers at smem_raw[0..72)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'reverse_pipe' ---
            // prep_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // smem_free: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // inputs_ready: 1 barriers, init_count=4
            mbarrier_init(smem + 16, 4);
            // dr_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // dr_inp_ready: 1 barriers, init_count=4
            mbarrier_init(smem + 32, 4);
            // dx_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // de_inp_ready: 1 barriers, init_count=4
            mbarrier_init(smem + 48, 4);
            // dh_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // compute_done: 1 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 72);
    if (warp == 0) {
        int _tmem_hold = smem + 72;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define prep_ready_addr (mbar_base + 0)
    #define smem_free_addr (mbar_base + 8)
    #define inputs_ready_addr (mbar_base + 16)
    #define dr_ready_addr (mbar_base + 24)
    #define dr_inp_ready_addr (mbar_base + 32)
    #define dx_ready_addr (mbar_base + 40)
    #define de_inp_ready_addr (mbar_base + 48)
    #define dh_ready_addr (mbar_base + 56)
    #define compute_done_addr (mbar_base + 64)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_dh = taddr + 64;
    const int tmem_tmem_dh_inp = taddr;
    const int tmem_tmem_do_initial = taddr + 224;
    const int tmem_tmem_do_final = taddr;
    const int tmem_tmem_dr = taddr + 192;
    const int tmem_tmem_dr_inp = taddr + 192;
    const int tmem_tmem_dx = taddr + 224;
    const int tmem_tmem_de_inp = taddr + 224;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
    }

    // ---- Role: compute ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // compute_main
            int split_task_idx = blockIdx.x;
            int task_idx = split_task_idx / 2;
            int value_split_idx = split_task_idx % 2;
            int value_row_offset = value_split_idx * 64;
            int seq_idx = seq_order[task_idx / num_heads];
            int head_idx = task_idx % num_heads;
            long long bos = cu_seqlens[seq_idx];
            long long eos = cu_seqlens[seq_idx + 1];
            int seq_len = (int)(eos - bos);
            int num_chunks = (seq_len + 32 - 1) / 32;
            int lane_quad = lane & 3;
            int warp_in_wg = warp % 4;
            int local_row_top = warp_in_wg * 16 + lane / 4;
            int local_row_bot = local_row_top + 8;
            int state_row_top = value_row_offset + local_row_top;
            int state_row_bot = value_row_offset + local_row_bot;
            const int tmem_row_base = warp_in_wg * 32 << 16;
            long long state_head_base = ((long long)seq_idx * (long long)num_heads + (long long)head_idx) * 128 * 128;
            long long state_base_top = state_head_base + (long long)state_row_top * 128;
            long long state_base_bot = state_head_base + (long long)state_row_bot * 128;
            #pragma unroll
            for (int state_col_half = 0; state_col_half < 2; state_col_half++) {
                float state_values[32];
                #pragma unroll
                for (int state_col_group = 0; state_col_group < 8; state_col_group++) {
                    int state_col_pair = state_col_half * 64 + state_col_group * 8 + lane_quad * 2;
                    const int state_reg_base = state_col_group * 4;
                    state_values[state_reg_base] = dfinal_state[state_base_top + (long long)state_col_pair];
                    state_values[state_reg_base + 1] = dfinal_state[state_base_top + (long long)state_col_pair + 1];
                    state_values[state_reg_base + 2] = dfinal_state[state_base_bot + (long long)state_col_pair];
                    state_values[state_reg_base + 3] = dfinal_state[state_base_bot + (long long)state_col_pair + 1];
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x8.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                    :: "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_half * 64)), "r"(*reinterpret_cast<const uint32_t*>(&state_values[0])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[1])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[2])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[3])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[4])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[5])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[6])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[7])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[8])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[9])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[10])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[11])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[12])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[13])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[14])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[15])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[16])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[17])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[18])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[19])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[20])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[21])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[22])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[23])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[24])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[25])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[26])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[27])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[28])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[29])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[30])), "r"(*reinterpret_cast<const uint32_t*>(&state_values[31]))
                    : "memory");
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int compute_stage = 0;
            float _exp2_0 = approx_exp2(lower_bound * 1.4426950408889634f * 16.0f);
            float common_factor = _exp2_0;
            unsigned int _phase_prep_ready = 0;
            unsigned int _phase_dr_ready = 0;
            unsigned int _phase_dx_ready = 0;
            unsigned int _phase_dh_ready = 0;
            #pragma unroll 1
            for (int reverse_chunk = 0; reverse_chunk < num_chunks; reverse_chunk++) {
                mbarrier_wait(prep_ready_addr + (compute_stage) * 8, _phase_prep_ready);
                int chunk_idx = num_chunks - 1 - reverse_chunk;
                long long chunk_global = cu_chunk_offsets[seq_idx] + (long long)chunk_idx;
                long long chunk_start = bos + (long long)chunk_idx * 32;
                float do_values[16];
                do_values[0] = 0.0f;
                do_values[1] = 0.0f;
                do_values[2] = 0.0f;
                do_values[3] = 0.0f;
                do_values[4] = 0.0f;
                do_values[5] = 0.0f;
                do_values[6] = 0.0f;
                do_values[7] = 0.0f;
                do_values[8] = 0.0f;
                do_values[9] = 0.0f;
                do_values[10] = 0.0f;
                do_values[11] = 0.0f;
                do_values[12] = 0.0f;
                do_values[13] = 0.0f;
                do_values[14] = 0.0f;
                do_values[15] = 0.0f;
                #pragma unroll
                for (int token_group = 0; token_group < 4; token_group++) {
                    int token_pair = token_group * 8 + lane_quad * 2;
                    const int token_reg_base = token_group * 4;
                    long long token0 = chunk_start + (long long)token_pair;
                    long long token1 = token0 + 1;
                    if (token0 < eos) {
                        long long token_head0 = (token0 * (long long)num_heads + (long long)head_idx) * 128;
                        do_values[token_reg_base] = (float)do_[token_head0 + (long long)state_row_top];
                        do_values[token_reg_base + 2] = (float)do_[token_head0 + (long long)state_row_bot];
                    }
                    if (token1 < eos) {
                        long long token_head1 = (token1 * (long long)num_heads + (long long)head_idx) * 128;
                        do_values[token_reg_base + 1] = (float)do_[token_head1 + (long long)state_row_top];
                        do_values[token_reg_base + 3] = (float)do_[token_head1 + (long long)state_row_bot];
                    }
                }
                uint32_t do_values_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(do_values[_lp*2 + 0], do_values[_lp*2+1 + 0]));
                    do_values_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(taddr + 224 + (unsigned int)tmem_row_base), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[7]))
                    : "memory");
                long long chunk_state_head_base = (chunk_global * (long long)num_heads + (long long)head_idx) * 128 * 128;
                long long chunk_dh_base_top = chunk_state_head_base + (long long)state_row_top * 128;
                long long chunk_dh_base_bot = chunk_state_head_base + (long long)state_row_bot * 128;
                long long restore_base = (chunk_global * (long long)num_heads + (long long)head_idx) * 128;
                #pragma unroll
                for (int state_col_half2 = 0; state_col_half2 < 2; state_col_half2++) {
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                        : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_half2 * 64))
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int state_col_group2 = 0; state_col_group2 < 8; state_col_group2++) {
                        int state_col_pair2 = state_col_half2 * 64 + state_col_group2 * 8 + lane_quad * 2;
                        const int state_reg_base2 = state_col_group2 * 4;
                        chunk_dh[chunk_dh_base_top + (long long)state_col_pair2] = _tmem_load_0[state_reg_base2];
                        chunk_dh[chunk_dh_base_top + (long long)state_col_pair2 + 1] = _tmem_load_0[state_reg_base2 + 1];
                        chunk_dh[chunk_dh_base_bot + (long long)state_col_pair2] = _tmem_load_0[state_reg_base2 + 2];
                        chunk_dh[chunk_dh_base_bot + (long long)state_col_pair2 + 1] = _tmem_load_0[state_reg_base2 + 3];
                    }
                    uint32_t _tmem_load_0_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + (unsigned int)tmem_row_base + (unsigned int)(state_col_half2 * 32)), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[15]))
                        : "memory");
                    #pragma unroll
                    for (int restore_group = 0; restore_group < 8; restore_group++) {
                        int restore_col_pair = state_col_half2 * 64 + restore_group * 8 + lane_quad * 2;
                        const int restore_reg_base = restore_group * 4;
                        float restore0 = tape_restore_factor[restore_base + (long long)restore_col_pair];
                        float restore1 = tape_restore_factor[restore_base + (long long)restore_col_pair + 1];
                        _tmem_load_0[restore_reg_base] = _tmem_load_0[restore_reg_base] * restore0;
                        _tmem_load_0[restore_reg_base + 1] = _tmem_load_0[restore_reg_base + 1] * restore1;
                        _tmem_load_0[restore_reg_base + 2] = _tmem_load_0[restore_reg_base + 2] * restore0;
                        _tmem_load_0[restore_reg_base + 3] = _tmem_load_0[restore_reg_base + 3] * restore1;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x256b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_half2 * 64)), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[31]))
                        : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(inputs_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(dr_ready_addr + (compute_stage) * 8, _phase_dr_ready);
                float _tmem_load_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                    : "r"(taddr + 192 + (unsigned int)tmem_row_base)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                long long chunk_value_head_base = (chunk_global * (long long)num_heads + (long long)head_idx) * 128 * 32;
                long long dr_tape_base_top = chunk_value_head_base + (long long)state_row_top * 32;
                long long dr_tape_base_bot = chunk_value_head_base + (long long)state_row_bot * 32;
                #pragma unroll
                for (int dr_group = 0; dr_group < 4; dr_group++) {
                    int dr_col_pair = dr_group * 8 + lane_quad * 2;
                    const int dr_reg_base = dr_group * 4;
                    chunk_dr[dr_tape_base_top + (long long)dr_col_pair] = _tmem_load_1[dr_reg_base];
                    chunk_dr[dr_tape_base_top + (long long)dr_col_pair + 1] = _tmem_load_1[dr_reg_base + 1];
                    chunk_dr[dr_tape_base_bot + (long long)dr_col_pair] = _tmem_load_1[dr_reg_base + 2];
                    chunk_dr[dr_tape_base_bot + (long long)dr_col_pair + 1] = _tmem_load_1[dr_reg_base + 3];
                }
                uint32_t _tmem_load_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(taddr + 192 + (unsigned int)tmem_row_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1_bf16[7]))
                    : "memory");
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(taddr + (unsigned int)tmem_row_base), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&do_values_bf16[7]))
                    : "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(dr_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(dx_ready_addr + (compute_stage) * 8, _phase_dx_ready);
                float _tmem_load_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                    : "r"(taddr + 224 + (unsigned int)tmem_row_base)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int dx_group = 0; dx_group < 4; dx_group++) {
                    int dx_col_pair = dx_group * 8 + lane_quad * 2;
                    const int dx_reg_base = dx_group * 4;
                    chunk_dx[dr_tape_base_top + (long long)dx_col_pair] = _tmem_load_2[dx_reg_base];
                    chunk_dx[dr_tape_base_top + (long long)dx_col_pair + 1] = _tmem_load_2[dx_reg_base + 1];
                    chunk_dx[dr_tape_base_bot + (long long)dx_col_pair] = _tmem_load_2[dx_reg_base + 2];
                    chunk_dx[dr_tape_base_bot + (long long)dx_col_pair + 1] = _tmem_load_2[dx_reg_base + 3];
                    float beta0 = 0.0f;
                    float beta1 = 0.0f;
                    long long beta_token0 = chunk_start + (long long)dx_col_pair;
                    long long beta_token1 = beta_token0 + 1;
                    if (beta_token0 < eos) {
                        beta0 = beta_active[beta_token0 * (long long)num_heads + (long long)head_idx];
                    }
                    if (beta_token1 < eos) {
                        beta1 = beta_active[beta_token1 * (long long)num_heads + (long long)head_idx];
                    }
                    _tmem_load_2[dx_reg_base] = _tmem_load_2[dx_reg_base] * (-beta0);
                    _tmem_load_2[dx_reg_base + 1] = _tmem_load_2[dx_reg_base + 1] * (-beta1);
                    _tmem_load_2[dx_reg_base + 2] = _tmem_load_2[dx_reg_base + 2] * (-beta0);
                    _tmem_load_2[dx_reg_base + 3] = _tmem_load_2[dx_reg_base + 3] * (-beta1);
                }
                uint32_t _tmem_load_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(taddr + 224 + (unsigned int)tmem_row_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2_bf16[7]))
                    : "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(de_inp_ready_addr + (compute_stage) * 8);
                }
                if (publish_ready != 0) {
                    __threadfence();
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            {
                                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(boundary_ready) + (chunk_global * (long long)num_heads + (long long)head_idx);
                                unsigned int _gc_old;
                                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                            }
                        }
                    }
                }
                mbarrier_wait(dh_ready_addr + (compute_stage) * 8, _phase_dh_ready);
                #pragma unroll
                for (int state_col_half3 = 0; state_col_half3 < 2; state_col_half3++) {
                    float _tmem_load_3[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                        : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_half3 * 64))
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_0 = {common_factor, common_factor};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x256b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_half3 * 64)), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[31]))
                        : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                compute_stage += 1;
                if (compute_stage == 1) { compute_stage = 0; _phase_prep_ready ^= 1; _phase_dr_ready ^= 1; _phase_dx_ready ^= 1; _phase_dh_ready ^= 1; }
            }
            #pragma unroll
            for (int final_half = 0; final_half < 2; final_half++) {
                float _tmem_load_4[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                    : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(final_half * 64))
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int final_group = 0; final_group < 8; final_group++) {
                    int final_col_pair = final_half * 64 + final_group * 8 + lane_quad * 2;
                    const int final_reg_base = final_group * 4;
                    dinitial_state[state_base_top + (long long)final_col_pair] = _tmem_load_4[final_reg_base];
                    dinitial_state[state_base_top + (long long)final_col_pair + 1] = _tmem_load_4[final_reg_base + 1];
                    dinitial_state[state_base_bot + (long long)final_col_pair] = _tmem_load_4[final_reg_base + 2];
                    dinitial_state[state_base_bot + (long long)final_col_pair + 1] = _tmem_load_4[final_reg_base + 3];
                }
            }
            if (elect_sync()) {
                mbarrier_arrive(compute_done_addr);
            }
        }
    }
    // ---- Role: prep ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // prep_main
            int split_task_idx_1 = blockIdx.x;
            int task_idx_1 = split_task_idx_1 / 2;
            int seq_idx_1 = seq_order[task_idx_1 / num_heads];
            int head_idx_1 = task_idx_1 % num_heads;
            long long bos_1 = cu_seqlens[seq_idx_1];
            long long eos_1 = cu_seqlens[seq_idx_1 + 1];
            int seq_len_1 = (int)(eos_1 - bos_1);
            int num_chunks_1 = (seq_len_1 + 32 - 1) / 32;
            int warp_id_in_role = (warp - 4);
            int prep_tid = warp_id_in_role * 32 + lane;
            int prep_local_warp = warp_id_in_role;
            unsigned int prep_stage = 0;
            unsigned int _phase_smem_free = 1;
            #pragma unroll 1
            for (int reverse_chunk_1 = 0; reverse_chunk_1 < num_chunks_1; reverse_chunk_1++) {
                mbarrier_wait(smem_free_addr + (prep_stage) * 8, _phase_smem_free);
                int chunk_idx_1 = num_chunks_1 - 1 - reverse_chunk_1;
                long long chunk_global_1 = cu_chunk_offsets[seq_idx_1] + (long long)chunk_idx_1;
                long long tape_vec_base = (chunk_global_1 * (long long)num_heads + (long long)head_idx_1) * 32 * 128;
                long long restore_base_1 = (chunk_global_1 * (long long)num_heads + (long long)head_idx_1) * 128;
                #pragma unroll
                for (int load_pass = 0; load_pass < 4; load_pass++) {
                    int item = load_pass * 128 + prep_tid;
                    int row = item / 16;
                    int segment = item % 16;
                    long long index = tape_vec_base + (long long)row * 128 + (long long)(segment * 8);
                    float qd_values[8];
                    float kd_values[8];
                    float kr_values[8];
                    float ki_values[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(tape_qd + index);
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&qd_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&qd_values[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(tape_kd + index);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&kd_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&kd_values[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(tape_kr + index);
                        uint4 _vld_2[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_2[_blk] = _vptr_2[_blk];
                            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&kr_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&kr_values[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_2[_pair]));
                            }
                        }
                    }
                    #pragma unroll
                    for (int elem = 0; elem < 8; elem++) {
                        float factor = tape_restore_factor[restore_base_1 + (long long)(segment * 8) + (long long)elem];
                        ki_values[elem] = kr_values[elem] / factor;
                    }
                    unsigned int packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_values[_lp*2 + 0], qd_values[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word * 4)), "r"((packed[word])));
                    }
                    unsigned int packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_values[_lp*2 + 0], kd_values[_lp*2+1 + 0]));
                        packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_1 = 0; word_1 < 4; word_1++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_1 * 4)), "r"((packed_0[word_1])));
                    }
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kr_values[_lp*2 + 0], kr_values[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_2 = 0; word_2 < 4; word_2++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_addr + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_2 * 4)), "r"((packed_1[word_2])));
                    }
                    unsigned int packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_values[_lp*2 + 0], ki_values[_lp*2+1 + 0]));
                        packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_3 = 0; word_3 < 4; word_3++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_ki_addr + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_3 * 4)), "r"((packed_2[word_3])));
                    }
                }
                int j_row = prep_tid / 4;
                int j_segment = prep_tid % 4;
                float j_values[8];
                long long j_base = (chunk_global_1 * (long long)num_heads + (long long)head_idx_1) * 32 * 32;
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(tape_j + j_base + (long long)j_row * 32 + (long long)(j_segment * 8));
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&j_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&j_values[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_3[_pair]));
                        }
                    }
                }
                unsigned int packed_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(j_values[_lp*2 + 0], j_values[_lp*2+1 + 0]));
                    packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int word_4 = 0; word_4 < 4; word_4++) {
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_j_addr + (unsigned int)(j_segment * 8 / 16 * 1024 + j_row * 32 + j_segment * 8 % 16 * 2 ^ (j_segment * 8 / 16 * 1024 + j_row * 32 + j_segment * 8 % 16 * 2 >> 7 & 1) << 4)) + (unsigned int)(word_4 * 4)), "r"((packed_3[word_4])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int row_base = prep_local_warp / 2 * 16;
                int col_base = prep_local_warp % 2 * 16;
                unsigned int a_frag[4];
                unsigned int b_frag[4];
                float acc[8];
                if (row_base <= col_base) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)(((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)(((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)(((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + (unsigned int)(((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                } else {
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                }
                int row0 = row_base + lane / 4;
                int row1 = row0 + 8;
                int col0 = col_base + lane % 4 * 2;
                float values[8];
                values[0] = 0.0f;
                values[1] = 0.0f;
                values[2] = 0.0f;
                values[3] = 0.0f;
                values[4] = 0.0f;
                values[5] = 0.0f;
                values[6] = 0.0f;
                values[7] = 0.0f;
                if (row0 <= col0) {
                    values[0] = acc[0];
                }
                if (row0 <= col0 + 1) {
                    values[1] = acc[1];
                }
                if (row1 <= col0) {
                    values[2] = acc[2];
                }
                if (row1 <= col0 + 1) {
                    values[3] = acc[3];
                }
                if (row0 <= col0 + 8) {
                    values[4] = acc[4];
                }
                if (row0 <= col0 + 9) {
                    values[5] = acc[5];
                }
                if (row1 <= col0 + 8) {
                    values[6] = acc[6];
                }
                if (row1 <= col0 + 9) {
                    values[7] = acc[7];
                }
                unsigned int packed_0_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    packed_0_1[_lp] = *(uint32_t*)&_bf2;
                }
                int lane_row = lane % 16;
                int lane_col = lane / 16 * 8;
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)(smem_n_addr + (unsigned int)((col_base + lane_col) / 16 * 1024 + (row_base + lane_row) * 32 + (col_base + lane_col) % 16 * 2 ^ ((col_base + lane_col) / 16 * 1024 + (row_base + lane_row) * 32 + (col_base + lane_col) % 16 * 2 >> 7 & 1) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[3]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (prep_local_warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(prep_ready_addr + (prep_stage) * 8);
                    }
                }
                prep_stage += 1;
                if (prep_stage == 1) { prep_stage = 0; _phase_smem_free ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            int split_task_idx_2 = blockIdx.x;
            int task_idx_2 = split_task_idx_2 / 2;
            int seq_idx_2 = seq_order[task_idx_2 / num_heads];
            long long eos_2 = cu_seqlens[seq_idx_2 + 1];
            long long bos_2 = cu_seqlens[seq_idx_2];
            int seq_len_2 = (int)(eos_2 - bos_2);
            int num_chunks_2 = (seq_len_2 + 32 - 1) / 32;
            unsigned int mma_stage = 0;
            unsigned int _phase_prep_ready_1 = 0;
            unsigned int _phase_inputs_ready = 0;
            unsigned int _phase_dr_inp_ready = 0;
            unsigned int _phase_de_inp_ready = 0;
            #pragma unroll 1
            for (int _reverse_chunk = 0; _reverse_chunk < num_chunks_2; _reverse_chunk++) {
                mbarrier_wait(prep_ready_addr + (mma_stage) * 8, _phase_prep_ready_1);
                mbarrier_wait(inputs_ready_addr + (mma_stage) * 8, _phase_inputs_ready);
                int _mma_b_lo_0 = make_warp_uniform(((smem_n_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 67634320;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dr), "r"(_mma_b_lo_0), "r"(tmem_tmem_do_initial), "r"(0));
                int _mma_b_lo_1 = make_warp_uniform(((smem_kr_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 67634320;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dr), "r"(_mma_b_lo_1), "r"(tmem_tmem_dh_inp), "r"(1));
                elect_commit(dr_ready_addr + (mma_stage) * 8);
                mbarrier_wait(dr_inp_ready_addr + (mma_stage) * 8, _phase_dr_inp_ready);
                int _mma_b_lo_2 = make_warp_uniform((((smem_j_trans_addr) >> 4) & 0x3FFF) | 0x400000);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 67699856;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dx), "r"(_mma_b_lo_2), "r"(tmem_tmem_dr_inp), "r"(0));
                elect_commit(dx_ready_addr + (mma_stage) * 8);
                mbarrier_wait(de_inp_ready_addr + (mma_stage) * 8, _phase_de_inp_ready);
                int _mma_b_lo_3 = make_warp_uniform((((smem_qd_trans_addr) >> 4) & 0x3FFF) | 0x1000000);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dh), "r"(_mma_b_lo_3), "r"(tmem_tmem_do_final), "r"(1));
                int _mma_b_lo_4 = make_warp_uniform((((smem_kd_trans_addr) >> 4) & 0x3FFF) | 0x1000000);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dh), "r"(_mma_b_lo_4), "r"(tmem_tmem_de_inp), "r"(1));
                elect_commit2(dh_ready_addr + (mma_stage) * 8, smem_free_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 1) { mma_stage = 0; _phase_prep_ready_1 ^= 1; _phase_inputs_ready ^= 1; _phase_dr_inp_ready ^= 1; _phase_de_inp_ready ^= 1; }
            }
            unsigned int _phase_compute_done_0 = 0;
            mbarrier_wait(compute_done_addr, _phase_compute_done_0);
            _phase_compute_done_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }

    // Cleanup
}

} // extern "C"

#undef FLASHKDA_INF
#undef NUM_REVERSE_PIPE_STAGES
#undef SMEM_SMEM_J_OFF
#undef SMEM_SMEM_J_STAGE_BYTES
#undef SMEM_SMEM_J_STRIDE
#undef SMEM_SMEM_J_TRANS_OFF
#undef SMEM_SMEM_J_TRANS_STAGE_BYTES
#undef SMEM_SMEM_J_TRANS_STRIDE
#undef SMEM_SMEM_KD_OFF
#undef SMEM_SMEM_KD_STAGE_BYTES
#undef SMEM_SMEM_KD_STRIDE
#undef SMEM_SMEM_KD_TRANS_OFF
#undef SMEM_SMEM_KD_TRANS_STAGE_BYTES
#undef SMEM_SMEM_KD_TRANS_STRIDE
#undef SMEM_SMEM_KI_OFF
#undef SMEM_SMEM_KI_STAGE_BYTES
#undef SMEM_SMEM_KI_STRIDE
#undef SMEM_SMEM_KR_OFF
#undef SMEM_SMEM_KR_STAGE_BYTES
#undef SMEM_SMEM_KR_STRIDE
#undef SMEM_SMEM_N_OFF
#undef SMEM_SMEM_N_STAGE_BYTES
#undef SMEM_SMEM_N_STRIDE
#undef SMEM_SMEM_QD_OFF
#undef SMEM_SMEM_QD_STAGE_BYTES
#undef SMEM_SMEM_QD_STRIDE
#undef SMEM_SMEM_QD_TRANS_OFF
#undef SMEM_SMEM_QD_TRANS_STAGE_BYTES
#undef SMEM_SMEM_QD_TRANS_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_NCOLS
#undef TMEM_TMEM_DE_INP_OFFSET
#undef TMEM_TMEM_DH_INP_OFFSET
#undef TMEM_TMEM_DH_OFFSET
#undef TMEM_TMEM_DO_FINAL_OFFSET
#undef TMEM_TMEM_DO_INITIAL_OFFSET
#undef TMEM_TMEM_DR_INP_OFFSET
#undef TMEM_TMEM_DR_OFFSET
#undef TMEM_TMEM_DX_OFFSET
#undef compute_done_addr
#undef de_inp_ready_addr
#undef dh_ready_addr
#undef dr_inp_ready_addr
#undef dr_ready_addr
#undef dx_ready_addr
#undef inputs_ready_addr
#undef prep_ready_addr
#undef smem_free_addr
#undef smem_j_addr
#undef smem_j_trans_addr
#undef smem_kd_addr
#undef smem_kd_trans_addr
#undef smem_ki_addr
#undef smem_kr_addr
#undef smem_n_addr
#undef smem_qd_addr
#undef smem_qd_trans_addr

#define FLASHKDA_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_TMEM_DH_OFFSET 64
#define TMEM_TMEM_DH_INP_OFFSET 0
#define TMEM_TMEM_DO_INITIAL_OFFSET 224
#define TMEM_TMEM_DO_FINAL_OFFSET 0
#define TMEM_TMEM_DR_OFFSET 192
#define TMEM_TMEM_DR_INP_OFFSET 192
#define TMEM_TMEM_DX_OFFSET 224
#define TMEM_TMEM_DE_INP_OFFSET 224
#define NUM_PREP_PIPE_STAGES 2
#define NUM_REVERSE_PIPE_STAGES 1
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 8192
#define SMEM_SMEM_QD_STRIDE 36864
#define SMEM_SMEM_QD_TRANS_OFF 1024
#define SMEM_SMEM_QD_TRANS_STAGE_BYTES 8192
#define SMEM_SMEM_QD_TRANS_STRIDE 36864
#define SMEM_SMEM_KD_OFF 9216
#define SMEM_SMEM_KD_STAGE_BYTES 8192
#define SMEM_SMEM_KD_STRIDE 36864
#define SMEM_SMEM_KD_TRANS_OFF 9216
#define SMEM_SMEM_KD_TRANS_STAGE_BYTES 8192
#define SMEM_SMEM_KD_TRANS_STRIDE 36864
#define SMEM_SMEM_KR_OFF 17408
#define SMEM_SMEM_KR_STAGE_BYTES 8192
#define SMEM_SMEM_KR_STRIDE 36864
#define SMEM_SMEM_KI_OFF 25600
#define SMEM_SMEM_KI_STAGE_BYTES 8192
#define SMEM_SMEM_KI_STRIDE 36864
#define SMEM_SMEM_J_OFF 33792
#define SMEM_SMEM_J_STAGE_BYTES 2048
#define SMEM_SMEM_J_STRIDE 36864
#define SMEM_SMEM_J_TRANS_OFF 33792
#define SMEM_SMEM_J_TRANS_STAGE_BYTES 2048
#define SMEM_SMEM_J_TRANS_STRIDE 36864
#define SMEM_SMEM_N_OFF 35840
#define SMEM_SMEM_N_STAGE_BYTES 2048
#define SMEM_SMEM_N_STRIDE 36864
#define SMEM_TOTAL 74752
#define THREADS 512

extern "C" {

__global__ __launch_bounds__(512) void
kernel_flashkda_backward_boundary_c32_tcgen(__nv_bfloat16* __restrict__ do_, float* __restrict__ dfinal_state, float* __restrict__ beta_active, long long* __restrict__ cu_seqlens, long long* __restrict__ cu_chunk_offsets, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ tape_qd, __nv_bfloat16* __restrict__ tape_kd, __nv_bfloat16* __restrict__ tape_kr, __nv_bfloat16* __restrict__ tape_j, float* __restrict__ tape_restore_factor, __nv_bfloat16* __restrict__ chunk_dh, __nv_bfloat16* __restrict__ chunk_dr, __nv_bfloat16* __restrict__ chunk_dx, unsigned int* __restrict__ boundary_ready, float* __restrict__ dinitial_state, int num_heads, int publish_ready, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* smem_qd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_qd_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_trans_addr = smem + 1024;
    __nv_bfloat16* smem_kd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kd_addr = smem + 9216;
    __nv_bfloat16* smem_kd_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kd_trans_addr = smem + 9216;
    __nv_bfloat16* smem_kr = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_kr_addr = smem + 17408;
    __nv_bfloat16* smem_ki = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int smem_ki_addr = smem + 25600;
    __nv_bfloat16* smem_j = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_j_addr = smem + 33792;
    __nv_bfloat16* smem_j_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_j_trans_addr = smem + 33792;
    __nv_bfloat16* smem_n = reinterpret_cast<__nv_bfloat16*>(smem_raw + 35840);
    const int smem_n_addr = smem + 35840;

    // Mbarrier init (8 groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'prep_pipe' ---
            // smem_free: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // prep_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'reverse_pipe' ---
            // dr_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // dr_inp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // dx_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // de_inp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // dh_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // owners_done: 1 barriers, init_count=12
            mbarrier_init(smem + 72, 12);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 0) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define smem_free_addr (mbar_base + 0)
    #define prep_ready_addr (mbar_base + 16)
    #define dr_ready_addr (mbar_base + 32)
    #define dr_inp_ready_addr (mbar_base + 40)
    #define dx_ready_addr (mbar_base + 48)
    #define de_inp_ready_addr (mbar_base + 56)
    #define dh_ready_addr (mbar_base + 64)
    #define owners_done_addr (mbar_base + 72)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_dh = taddr + 64;
    const int tmem_tmem_dh_inp = taddr;
    const int tmem_tmem_do_initial = taddr + 224;
    const int tmem_tmem_do_final = taddr;
    const int tmem_tmem_dr = taddr + 192;
    const int tmem_tmem_dr_inp = taddr + 192;
    const int tmem_tmem_dx = taddr + 224;
    const int tmem_tmem_de_inp = taddr + 224;

    // ---- Role: state ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // state_main
            int task_idx = blockIdx.x;
            int seq_idx = seq_order[task_idx / num_heads];
            int head_idx = task_idx % num_heads;
            long long bos = cu_seqlens[seq_idx];
            long long eos = cu_seqlens[seq_idx + 1];
            int seq_len = (int)(eos - bos);
            int num_chunks = (seq_len + 32 - 1) / 32;
            int warp_id_in_role = (warp - 0);
            int state_row = warp_id_in_role * 32 + lane;
            int tmem_row_base = warp_id_in_role * 32 << 16;
            long long state_base = (((long long)seq_idx * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
            #pragma unroll
            for (int state_block = 0; state_block < 4; state_block++) {
                float state_values[32];
                #pragma unroll
                for (int state_vec = 0; state_vec < 4; state_vec++) {
                    {
                        unsigned _ldv8_0_0;
                        unsigned _ldv8_0_1;
                        unsigned _ldv8_0_2;
                        unsigned _ldv8_0_3;
                        unsigned _ldv8_0_4;
                        unsigned _ldv8_0_5;
                        unsigned _ldv8_0_6;
                        unsigned _ldv8_0_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(dfinal_state + (state_base + (long long)(state_block * 32) + (long long)(state_vec * 8)))) : "memory");
                        state_values[state_vec * 8 + 0] = __uint_as_float(_ldv8_0_0);
                        state_values[state_vec * 8 + 1] = __uint_as_float(_ldv8_0_1);
                        state_values[state_vec * 8 + 2] = __uint_as_float(_ldv8_0_2);
                        state_values[state_vec * 8 + 3] = __uint_as_float(_ldv8_0_3);
                        state_values[state_vec * 8 + 4] = __uint_as_float(_ldv8_0_4);
                        state_values[state_vec * 8 + 5] = __uint_as_float(_ldv8_0_5);
                        state_values[state_vec * 8 + 6] = __uint_as_float(_ldv8_0_6);
                        state_values[state_vec * 8 + 7] = __uint_as_float(_ldv8_0_7);
                    }
                }
                tmem_st_x32_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_block * 32), state_values);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int state_stage = 0;
            unsigned int issue_smem_stage = 0;
            float _exp2_0 = approx_exp2(lower_bound * 1.4426950408889634f * 16.0f);
            float common_factor = _exp2_0;
            unsigned int _phase_prep_ready = 0;
            unsigned int _phase_dr_inp_ready = 0;
            unsigned int _phase_de_inp_ready = 0;
            unsigned int _phase_dh_ready = 0;
            #pragma unroll 1
            for (int reverse_chunk = 0; reverse_chunk < num_chunks; reverse_chunk++) {
                int chunk_idx = num_chunks - 1 - reverse_chunk;
                long long chunk_global = cu_chunk_offsets[seq_idx] + (long long)chunk_idx;
                long long chunk_dh_base = ((chunk_global * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
                long long restore_base = (chunk_global * (long long)num_heads + (long long)head_idx) * 128;
                #pragma unroll
                for (int state_block2 = 0; state_block2 < 4; state_block2++) {
                    float _tmem_load_0[32];
                    tmem_ld_x32(&_tmem_load_0[0], taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_block2 * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (reverse_chunk != 0) {
                        const float2 _scale2_1 = {common_factor, common_factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_1);
                    }
                    #pragma unroll
                    for (int dh_store_vec = 0; dh_store_vec < 4; dh_store_vec++) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_0[dh_store_vec * 8 + 0], _tmem_load_0[dh_store_vec * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_0[dh_store_vec * 8 + 2], _tmem_load_0[dh_store_vec * 8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_0[dh_store_vec * 8 + 4], _tmem_load_0[dh_store_vec * 8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_0[dh_store_vec * 8 + 6], _tmem_load_0[dh_store_vec * 8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(chunk_dh + (chunk_dh_base + (long long)(state_block2 * 32) + (long long)(dh_store_vec * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    uint32_t _tmem_load_0_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + (unsigned int)tmem_row_base + (unsigned int)(state_block2 * 16)), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[15]))
                        : "memory");
                    float restore_lane = tape_restore_factor[restore_base + (long long)(state_block2 * 32) + (long long)lane];
                    #pragma unroll
                    for (int restore_elem = 0; restore_elem < 32; restore_elem++) {
                        float _shfl_0 = __shfl_sync(0xFFFFFFFF, restore_lane, restore_elem);
                        _tmem_load_0[restore_elem] = _tmem_load_0[restore_elem] * _shfl_0;
                    }
                    tmem_st_x32_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_block2 * 32), _tmem_load_0);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("barrier.sync 9, 384;" ::: "memory");
                if (warp_id_in_role == 0) {
                    mbarrier_wait(prep_ready_addr + (issue_smem_stage) * 8, _phase_prep_ready);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_n_addr) >> 4) & 0x3FFF) + (issue_smem_stage) * 2304);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dr), "r"(_mma_b_lo_0), "r"(tmem_tmem_do_initial), "r"(0));
                    int _mma_b_lo_1 = make_warp_uniform((((smem_kr_addr) >> 4) & 0x3FFF) + (issue_smem_stage) * 2304);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dr), "r"(_mma_b_lo_1), "r"(tmem_tmem_dh_inp), "r"(1));
                    elect_commit(dr_ready_addr + (state_stage) * 8);
                    mbarrier_wait(dr_inp_ready_addr + (state_stage) * 8, _phase_dr_inp_ready);
                    int _mma_b_lo_2 = make_warp_uniform(((((smem_j_trans_addr) >> 4) & 0x3FFF) | 0x400000) + (issue_smem_stage) * 2304);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134808720;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dx), "r"(_mma_b_lo_2), "r"(tmem_tmem_dr_inp), "r"(0));
                    elect_commit(dx_ready_addr + (state_stage) * 8);
                    mbarrier_wait(de_inp_ready_addr + (state_stage) * 8, _phase_de_inp_ready);
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_qd_trans_addr) >> 4) & 0x3FFF) | 0x1000000) + (issue_smem_stage) * 2304);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dh), "r"(_mma_b_lo_3), "r"(tmem_tmem_do_final), "r"(1));
                    int _mma_b_lo_4 = make_warp_uniform(((((smem_kd_trans_addr) >> 4) & 0x3FFF) | 0x1000000) + (issue_smem_stage) * 2304);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dh), "r"(_mma_b_lo_4), "r"(tmem_tmem_de_inp), "r"(1));
                    elect_commit2(dh_ready_addr + (state_stage) * 8, smem_free_addr + (issue_smem_stage) * 8);
                    issue_smem_stage += 1;
                    if (issue_smem_stage == 2) { issue_smem_stage = 0; _phase_prep_ready ^= 1; }
                }
                if (publish_ready != 0) {
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    if (warp_id_in_role == 0) {
                        if (elect_sync()) {
                            {
                                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(boundary_ready) + (chunk_global * (long long)num_heads + (long long)head_idx);
                                unsigned int _gc_old;
                                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                            }
                        }
                    }
                }
                mbarrier_wait(dh_ready_addr + (state_stage) * 8, _phase_dh_ready);
                state_stage += 1;
                if (state_stage == 1) { state_stage = 0; _phase_dr_inp_ready ^= 1; _phase_de_inp_ready ^= 1; _phase_dh_ready ^= 1; }
            }
            #pragma unroll
            for (int final_block = 0; final_block < 4; final_block++) {
                float _tmem_load_1[32];
                tmem_ld_x32(&_tmem_load_1[0], taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(final_block * 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_2 = {common_factor, common_factor};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_2);
                #pragma unroll
                for (int final_vec = 0; final_vec < 4; final_vec++) {
                    {
                        unsigned _stv8_3_0 = __float_as_uint(_tmem_load_1[final_vec * 8 + 0]);
                        unsigned _stv8_3_1 = __float_as_uint(_tmem_load_1[final_vec * 8 + 1]);
                        unsigned _stv8_3_2 = __float_as_uint(_tmem_load_1[final_vec * 8 + 2]);
                        unsigned _stv8_3_3 = __float_as_uint(_tmem_load_1[final_vec * 8 + 3]);
                        unsigned _stv8_3_4 = __float_as_uint(_tmem_load_1[final_vec * 8 + 4]);
                        unsigned _stv8_3_5 = __float_as_uint(_tmem_load_1[final_vec * 8 + 5]);
                        unsigned _stv8_3_6 = __float_as_uint(_tmem_load_1[final_vec * 8 + 6]);
                        unsigned _stv8_3_7 = __float_as_uint(_tmem_load_1[final_vec * 8 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dinitial_state + (state_base + (long long)(final_block * 32) + (long long)(final_vec * 8)) + (0))), "r"(_stv8_3_0), "r"(_stv8_3_1), "r"(_stv8_3_2), "r"(_stv8_3_3), "r"(_stv8_3_4), "r"(_stv8_3_5), "r"(_stv8_3_6), "r"(_stv8_3_7) : "memory");
                    }
                }
            }
            if (elect_sync()) {
                mbarrier_arrive(owners_done_addr);
            }
        }
    }
    // ---- Role: prep ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // prep_main
            int task_idx_1 = blockIdx.x;
            int seq_idx_1 = seq_order[task_idx_1 / num_heads];
            int head_idx_1 = task_idx_1 % num_heads;
            long long bos_1 = cu_seqlens[seq_idx_1];
            long long eos_1 = cu_seqlens[seq_idx_1 + 1];
            int seq_len_1 = (int)(eos_1 - bos_1);
            int num_chunks_1 = (seq_len_1 + 32 - 1) / 32;
            int warp_id_in_role_1 = (warp - 4);
            int prep_tid = warp_id_in_role_1 * 32 + lane;
            int prep_local_warp = warp_id_in_role_1;
            unsigned int prep_stage = 0;
            unsigned int _phase_smem_free = 1;
            #pragma unroll 1
            for (int reverse_chunk_1 = 0; reverse_chunk_1 < num_chunks_1; reverse_chunk_1++) {
                mbarrier_wait(smem_free_addr + (prep_stage) * 8, _phase_smem_free);
                int chunk_idx_1 = num_chunks_1 - 1 - reverse_chunk_1;
                long long chunk_global_1 = cu_chunk_offsets[seq_idx_1] + (long long)chunk_idx_1;
                long long tape_vec_base = (chunk_global_1 * (long long)num_heads + (long long)head_idx_1) * 32 * 128;
                long long restore_base_1 = (chunk_global_1 * (long long)num_heads + (long long)head_idx_1) * 128;
                #pragma unroll
                for (int load_pass = 0; load_pass < 4; load_pass++) {
                    int item = load_pass * 128 + prep_tid;
                    int row = item / 16;
                    int segment = item % 16;
                    long long index = tape_vec_base + (long long)row * 128 + (long long)(segment * 8);
                    float qd_values[8];
                    float kd_values[8];
                    float kr_values[8];
                    float ki_values[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(tape_qd + index);
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&qd_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&qd_values[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(tape_kd + index);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&kd_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&kd_values[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(tape_kr + index);
                        uint4 _vld_2[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_2[_blk] = _vptr_2[_blk];
                            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&kr_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&kr_values[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_2[_pair]));
                            }
                        }
                    }
                    #pragma unroll
                    for (int elem = 0; elem < 8; elem++) {
                        float factor = tape_restore_factor[restore_base_1 + (long long)(segment * 8) + (long long)elem];
                        ki_values[elem] = kr_values[elem] / factor;
                    }
                    unsigned int packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_values[_lp*2 + 0], qd_values[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + prep_stage * 36864 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word * 4)), "r"((packed[word])));
                    }
                    unsigned int packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_values[_lp*2 + 0], kd_values[_lp*2+1 + 0]));
                        packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_1 = 0; word_1 < 4; word_1++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 36864 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_1 * 4)), "r"((packed_0[word_1])));
                    }
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kr_values[_lp*2 + 0], kr_values[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_2 = 0; word_2 < 4; word_2++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_addr + prep_stage * 36864 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_2 * 4)), "r"((packed_1[word_2])));
                    }
                    unsigned int packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_values[_lp*2 + 0], ki_values[_lp*2+1 + 0]));
                        packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_3 = 0; word_3 < 4; word_3++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_ki_addr + prep_stage * 36864 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_3 * 4)), "r"((packed_2[word_3])));
                    }
                }
                int j_row = prep_tid / 4;
                int j_segment = prep_tid % 4;
                float j_values[8];
                long long j_base = (chunk_global_1 * (long long)num_heads + (long long)head_idx_1) * 32 * 32;
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(tape_j + j_base + (long long)j_row * 32 + (long long)(j_segment * 8));
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&j_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&j_values[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_3[_pair]));
                        }
                    }
                }
                unsigned int packed_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(j_values[_lp*2 + 0], j_values[_lp*2+1 + 0]));
                    packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int word_4 = 0; word_4 < 4; word_4++) {
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_j_addr + prep_stage * 36864 + (unsigned int)(j_segment * 8 / 16 * 1024 + j_row * 32 + j_segment * 8 % 16 * 2 ^ (j_segment * 8 / 16 * 1024 + j_row * 32 + j_segment * 8 % 16 * 2 >> 7 & 1) << 4)) + (unsigned int)(word_4 * 4)), "r"((packed_3[word_4])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int row_base = prep_local_warp / 2 * 16;
                int col_base = prep_local_warp % 2 * 16;
                unsigned int a_frag[4];
                unsigned int b_frag[4];
                float acc[8];
                if (row_base <= col_base) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 36864 + (unsigned int)((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 36864 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 36864 + (unsigned int)((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 36864 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 36864 + (unsigned int)((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 36864 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 36864 + (unsigned int)((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 36864 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 36864 + (unsigned int)(((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 36864 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 36864 + (unsigned int)(((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 36864 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 36864 + (unsigned int)(((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 36864 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"(smem_ki_addr + prep_stage * 36864 + (unsigned int)(((lane / 16 / 8 * 256 + (row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(smem_qd_addr + prep_stage * 36864 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                } else {
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                }
                int row0 = row_base + lane / 4;
                int row1 = row0 + 8;
                int col0 = col_base + lane % 4 * 2;
                float values[8];
                values[0] = 0.0f;
                values[1] = 0.0f;
                values[2] = 0.0f;
                values[3] = 0.0f;
                values[4] = 0.0f;
                values[5] = 0.0f;
                values[6] = 0.0f;
                values[7] = 0.0f;
                if (row0 <= col0) {
                    values[0] = acc[0];
                }
                if (row0 <= col0 + 1) {
                    values[1] = acc[1];
                }
                if (row1 <= col0) {
                    values[2] = acc[2];
                }
                if (row1 <= col0 + 1) {
                    values[3] = acc[3];
                }
                if (row0 <= col0 + 8) {
                    values[4] = acc[4];
                }
                if (row0 <= col0 + 9) {
                    values[5] = acc[5];
                }
                if (row1 <= col0 + 8) {
                    values[6] = acc[6];
                }
                if (row1 <= col0 + 9) {
                    values[7] = acc[7];
                }
                unsigned int packed_0_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    packed_0_1[_lp] = *(uint32_t*)&_bf2;
                }
                int lane_row = lane % 16;
                int lane_col = lane / 16 * 8;
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)(smem_n_addr + prep_stage * 36864 + (unsigned int)((col_base + lane_col) / 16 * 1024 + (row_base + lane_row) * 32 + (col_base + lane_col) % 16 * 2 ^ ((col_base + lane_col) / 16 * 1024 + (row_base + lane_row) * 32 + (col_base + lane_col) % 16 * 2 >> 7 & 1) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_0_1[3]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (prep_local_warp == 2) {
                    if (elect_sync()) {
                        mbarrier_arrive(prep_ready_addr + (prep_stage) * 8);
                    }
                }
                prep_stage += 1;
                if (prep_stage == 2) { prep_stage = 0; _phase_smem_free ^= 1; }
            }
            unsigned int _phase_owners_done_0 = 0;
            if (prep_local_warp == 2) {
                mbarrier_wait(owners_done_addr, _phase_owners_done_0);
                _phase_owners_done_0 ^= 1;
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
            }
        }
    }
    // ---- Role: high_token ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // high_token_main
            int task_idx_2 = blockIdx.x;
            int seq_idx_2 = seq_order[task_idx_2 / num_heads];
            int head_idx_2 = task_idx_2 % num_heads;
            long long bos_2 = cu_seqlens[seq_idx_2];
            long long eos_2 = cu_seqlens[seq_idx_2 + 1];
            int seq_len_2 = (int)(eos_2 - bos_2);
            int num_chunks_2 = (seq_len_2 + 32 - 1) / 32;
            const int token_offset = 16;
            unsigned int token_stage = 0;
            unsigned int _phase_dr_ready = 0;
            unsigned int _phase_dx_ready = 0;
            unsigned int _phase_dh_ready_1 = 0;
            #pragma unroll 1
            for (int reverse_chunk_2 = 0; reverse_chunk_2 < num_chunks_2; reverse_chunk_2++) {
                int chunk_idx_2 = num_chunks_2 - 1 - reverse_chunk_2;
                long long chunk_global_2 = cu_chunk_offsets[seq_idx_2] + (long long)chunk_idx_2;
                long long chunk_start = bos_2 + (long long)chunk_idx_2 * 32;
                float do_values[16];
                do_values[0] = 0.0f;
                do_values[1] = 0.0f;
                do_values[2] = 0.0f;
                do_values[3] = 0.0f;
                do_values[4] = 0.0f;
                do_values[5] = 0.0f;
                do_values[6] = 0.0f;
                do_values[7] = 0.0f;
                do_values[8] = 0.0f;
                do_values[9] = 0.0f;
                do_values[10] = 0.0f;
                do_values[11] = 0.0f;
                do_values[12] = 0.0f;
                do_values[13] = 0.0f;
                do_values[14] = 0.0f;
                do_values[15] = 0.0f;
                #pragma unroll
                for (int row_pass = 0; row_pass < 1; row_pass++) {
                    int warp_id_in_role_2 = (warp - 8);
                    int state_row_1 = (warp_id_in_role_2 + row_pass) * 32 + lane;
                    #pragma unroll
                    for (int token_local = 0; token_local < 16; token_local++) {
                        int token_col = token_offset + token_local;
                        long long token_idx = chunk_start + (long long)token_col;
                        if (token_idx < eos_2) {
                            do_values[row_pass * 16 + token_local] = (float)do_[(token_idx * (long long)num_heads + (long long)head_idx_2) * 128 + (long long)state_row_1];
                        }
                    }
                }
                uint32_t do_values_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(do_values[_lp*2 + 0], do_values[_lp*2+1 + 0]));
                    do_values_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int row_pass2 = 0; row_pass2 < 1; row_pass2++) {
                    int warp_id_in_role_3 = (warp - 8);
                    int row_group = warp_id_in_role_3 + row_pass2;
                    int tmem_row_base_1 = row_group * 32 << 16;
                    tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base_1 + (unsigned int)(token_offset / 2), (const uint32_t*)(do_values_bf16 + row_pass2 * 8));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("barrier.sync 9, 384;" ::: "memory");
                mbarrier_wait(dr_ready_addr + (token_stage) * 8, _phase_dr_ready);
                #pragma unroll
                for (int dr_row_pass = 0; dr_row_pass < 1; dr_row_pass++) {
                    int warp_id_in_role_4 = (warp - 8);
                    int row_group2 = warp_id_in_role_4 + dr_row_pass;
                    int state_row2 = row_group2 * 32 + lane;
                    int tmem_row_base2 = row_group2 * 32 << 16;
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + 192 + (unsigned int)tmem_row_base2 + (unsigned int)token_offset);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    long long dr_tape_base = ((chunk_global_2 * (long long)num_heads + (long long)head_idx_2) * 128 + (long long)state_row2) * 32;
                    #pragma unroll
                    for (int dr_store_vec = 0; dr_store_vec < 2; dr_store_vec++) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_4[dr_store_vec * 8 + 0], _tmem_load_4[dr_store_vec * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_4[dr_store_vec * 8 + 2], _tmem_load_4[dr_store_vec * 8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_4[dr_store_vec * 8 + 4], _tmem_load_4[dr_store_vec * 8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_4[dr_store_vec * 8 + 6], _tmem_load_4[dr_store_vec * 8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(chunk_dr + (dr_tape_base + (long long)token_offset + (long long)(dr_store_vec * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    uint32_t _tmem_load_4_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                        _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 192 + (unsigned int)tmem_row_base2 + (unsigned int)(token_offset / 2), (const uint32_t*)_tmem_load_4_bf16);
                    tmem_st_x8_u32(taddr + (unsigned int)tmem_row_base2 + (unsigned int)(token_offset / 2), (const uint32_t*)(do_values_bf16 + dr_row_pass * 8));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("barrier.sync 10, 256;" ::: "memory");
                mbarrier_wait(dx_ready_addr + (token_stage) * 8, _phase_dx_ready);
                long long beta_token = chunk_start + (long long)lane;
                float beta_lane = 0.0f;
                if (beta_token < eos_2) {
                    beta_lane = beta_active[beta_token * (long long)num_heads + (long long)head_idx_2];
                }
                #pragma unroll
                for (int dx_row_pass = 0; dx_row_pass < 1; dx_row_pass++) {
                    int warp_id_in_role_5 = (warp - 8);
                    int row_group3 = warp_id_in_role_5 + dx_row_pass;
                    int state_row3 = row_group3 * 32 + lane;
                    int tmem_row_base3 = row_group3 * 32 << 16;
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + 224 + (unsigned int)tmem_row_base3 + (unsigned int)token_offset);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    long long dx_tape_base = ((chunk_global_2 * (long long)num_heads + (long long)head_idx_2) * 128 + (long long)state_row3) * 32;
                    #pragma unroll
                    for (int dx_store_vec = 0; dx_store_vec < 2; dx_store_vec++) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_5[dx_store_vec * 8 + 0], _tmem_load_5[dx_store_vec * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_5[dx_store_vec * 8 + 2], _tmem_load_5[dx_store_vec * 8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_5[dx_store_vec * 8 + 4], _tmem_load_5[dx_store_vec * 8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_5[dx_store_vec * 8 + 6], _tmem_load_5[dx_store_vec * 8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(chunk_dx + (dx_tape_base + (long long)token_offset + (long long)(dx_store_vec * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    #pragma unroll
                    for (int token_local2 = 0; token_local2 < 16; token_local2++) {
                        int token_col2 = token_offset + token_local2;
                        float _shfl_2 = __shfl_sync(0xFFFFFFFF, beta_lane, token_col2);
                        float beta_value = _shfl_2;
                        _tmem_load_5[token_local2] = _tmem_load_5[token_local2] * (-beta_value);
                    }
                    uint32_t _tmem_load_5_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                        _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base3 + (unsigned int)(token_offset / 2), (const uint32_t*)_tmem_load_5_bf16);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("barrier.sync 11, 256;" ::: "memory");
                if (publish_ready != 0) {
                    asm volatile("barrier.sync 14, 128;" ::: "memory");
                    int warp_id_in_role_6 = (warp - 8);
                    if (warp_id_in_role_6 == 0) {
                        if (elect_sync()) {
                            {
                                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(boundary_ready) + (chunk_global_2 * (long long)num_heads + (long long)head_idx_2);
                                unsigned int _gc_old;
                                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                            }
                        }
                    }
                }
                mbarrier_wait(dh_ready_addr + (token_stage) * 8, _phase_dh_ready_1);
                token_stage += 1;
                if (token_stage == 1) { token_stage = 0; _phase_dr_ready ^= 1; _phase_dx_ready ^= 1; _phase_dh_ready_1 ^= 1; }
            }
            if (elect_sync()) {
                mbarrier_arrive(owners_done_addr);
            }
        }
    }
    // ---- Role: low_token ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // low_token_main
            int task_idx_3 = blockIdx.x;
            int seq_idx_3 = seq_order[task_idx_3 / num_heads];
            int head_idx_3 = task_idx_3 % num_heads;
            long long bos_3 = cu_seqlens[seq_idx_3];
            long long eos_3 = cu_seqlens[seq_idx_3 + 1];
            int seq_len_3 = (int)(eos_3 - bos_3);
            int num_chunks_3 = (seq_len_3 + 32 - 1) / 32;
            int warp_id_in_role_7 = (warp - 12);
            int state_row_2 = warp_id_in_role_7 * 32 + lane;
            int tmem_row_base_2 = warp_id_in_role_7 * 32 << 16;
            unsigned int low_token_stage = 0;
            unsigned int _phase_dr_ready_1 = 0;
            unsigned int _phase_dx_ready_1 = 0;
            unsigned int _phase_dh_ready_2 = 0;
            #pragma unroll 1
            for (int reverse_chunk_3 = 0; reverse_chunk_3 < num_chunks_3; reverse_chunk_3++) {
                int chunk_idx_3 = num_chunks_3 - 1 - reverse_chunk_3;
                long long chunk_global_3 = cu_chunk_offsets[seq_idx_3] + (long long)chunk_idx_3;
                long long chunk_start_1 = bos_3 + (long long)chunk_idx_3 * 32;
                float do_values_1[16];
                do_values_1[0] = 0.0f;
                do_values_1[1] = 0.0f;
                do_values_1[2] = 0.0f;
                do_values_1[3] = 0.0f;
                do_values_1[4] = 0.0f;
                do_values_1[5] = 0.0f;
                do_values_1[6] = 0.0f;
                do_values_1[7] = 0.0f;
                do_values_1[8] = 0.0f;
                do_values_1[9] = 0.0f;
                do_values_1[10] = 0.0f;
                do_values_1[11] = 0.0f;
                do_values_1[12] = 0.0f;
                do_values_1[13] = 0.0f;
                do_values_1[14] = 0.0f;
                do_values_1[15] = 0.0f;
                #pragma unroll
                for (int token_col_1 = 0; token_col_1 < 16; token_col_1++) {
                    long long token_idx_1 = chunk_start_1 + (long long)token_col_1;
                    if (token_idx_1 < eos_3) {
                        do_values_1[token_col_1] = (float)do_[(token_idx_1 * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)state_row_2];
                    }
                }
                uint32_t do_values_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(do_values_1[_lp*2 + 0], do_values_1[_lp*2+1 + 0]));
                    do_values_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base_2, (const uint32_t*)do_values_bf16_1);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("barrier.sync 9, 384;" ::: "memory");
                mbarrier_wait(dr_ready_addr + (low_token_stage) * 8, _phase_dr_ready_1);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + 192 + (unsigned int)tmem_row_base_2);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                long long dr_tape_base_1 = ((chunk_global_3 * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)state_row_2) * 32;
                #pragma unroll
                for (int dr_store_vec_1 = 0; dr_store_vec_1 < 2; dr_store_vec_1++) {
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_2[dr_store_vec_1 * 8 + 0], _tmem_load_2[dr_store_vec_1 * 8 + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_2[dr_store_vec_1 * 8 + 2], _tmem_load_2[dr_store_vec_1 * 8 + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_2[dr_store_vec_1 * 8 + 4], _tmem_load_2[dr_store_vec_1 * 8 + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_2[dr_store_vec_1 * 8 + 6], _tmem_load_2[dr_store_vec_1 * 8 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(chunk_dr + (dr_tape_base_1 + (long long)(dr_store_vec_1 * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
                uint32_t _tmem_load_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                tmem_st_x8_u32(taddr + 192 + (unsigned int)tmem_row_base_2, (const uint32_t*)_tmem_load_2_bf16);
                tmem_st_x8_u32(taddr + (unsigned int)tmem_row_base_2, (const uint32_t*)do_values_bf16_1);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("barrier.sync 10, 256;" ::: "memory");
                if (warp_id_in_role_7 == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(dr_inp_ready_addr + (low_token_stage) * 8);
                    }
                }
                mbarrier_wait(dx_ready_addr + (low_token_stage) * 8, _phase_dx_ready_1);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + 224 + (unsigned int)tmem_row_base_2);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int dx_store_vec_1 = 0; dx_store_vec_1 < 2; dx_store_vec_1++) {
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(_tmem_load_3[dx_store_vec_1 * 8 + 0], _tmem_load_3[dx_store_vec_1 * 8 + 1]);
                        _pk[1] = __floats2bfloat162_rn(_tmem_load_3[dx_store_vec_1 * 8 + 2], _tmem_load_3[dx_store_vec_1 * 8 + 3]);
                        _pk[2] = __floats2bfloat162_rn(_tmem_load_3[dx_store_vec_1 * 8 + 4], _tmem_load_3[dx_store_vec_1 * 8 + 5]);
                        _pk[3] = __floats2bfloat162_rn(_tmem_load_3[dx_store_vec_1 * 8 + 6], _tmem_load_3[dx_store_vec_1 * 8 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(chunk_dx + (dr_tape_base_1 + (long long)(dx_store_vec_1 * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
                long long beta_token_1 = chunk_start_1 + (long long)lane;
                float beta_lane_1 = 0.0f;
                if (beta_token_1 < eos_3) {
                    beta_lane_1 = beta_active[beta_token_1 * (long long)num_heads + (long long)head_idx_3];
                }
                #pragma unroll
                for (int token_col2_1 = 0; token_col2_1 < 16; token_col2_1++) {
                    float _shfl_1 = __shfl_sync(0xFFFFFFFF, beta_lane_1, token_col2_1);
                    float beta_value_1 = _shfl_1;
                    _tmem_load_3[token_col2_1] = _tmem_load_3[token_col2_1] * (-beta_value_1);
                }
                uint32_t _tmem_load_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base_2, (const uint32_t*)_tmem_load_3_bf16);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("barrier.sync 11, 256;" ::: "memory");
                if (warp_id_in_role_7 == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(de_inp_ready_addr + (low_token_stage) * 8);
                    }
                }
                if (publish_ready != 0) {
                    asm volatile("barrier.sync 13, 128;" ::: "memory");
                    if (warp_id_in_role_7 == 0) {
                        if (elect_sync()) {
                            {
                                unsigned int* _gc_p = reinterpret_cast<unsigned int*>(boundary_ready) + (chunk_global_3 * (long long)num_heads + (long long)head_idx_3);
                                unsigned int _gc_old;
                                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                            }
                        }
                    }
                }
                mbarrier_wait(dh_ready_addr + (low_token_stage) * 8, _phase_dh_ready_2);
                low_token_stage += 1;
                if (low_token_stage == 1) { low_token_stage = 0; _phase_dr_ready_1 ^= 1; _phase_dx_ready_1 ^= 1; _phase_dh_ready_2 ^= 1; }
            }
            if (elect_sync()) {
                mbarrier_arrive(owners_done_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef FLASHKDA_INF
#undef NUM_PREP_PIPE_STAGES
#undef NUM_REVERSE_PIPE_STAGES
#undef SMEM_SMEM_J_OFF
#undef SMEM_SMEM_J_STAGE_BYTES
#undef SMEM_SMEM_J_STRIDE
#undef SMEM_SMEM_J_TRANS_OFF
#undef SMEM_SMEM_J_TRANS_STAGE_BYTES
#undef SMEM_SMEM_J_TRANS_STRIDE
#undef SMEM_SMEM_KD_OFF
#undef SMEM_SMEM_KD_STAGE_BYTES
#undef SMEM_SMEM_KD_STRIDE
#undef SMEM_SMEM_KD_TRANS_OFF
#undef SMEM_SMEM_KD_TRANS_STAGE_BYTES
#undef SMEM_SMEM_KD_TRANS_STRIDE
#undef SMEM_SMEM_KI_OFF
#undef SMEM_SMEM_KI_STAGE_BYTES
#undef SMEM_SMEM_KI_STRIDE
#undef SMEM_SMEM_KR_OFF
#undef SMEM_SMEM_KR_STAGE_BYTES
#undef SMEM_SMEM_KR_STRIDE
#undef SMEM_SMEM_N_OFF
#undef SMEM_SMEM_N_STAGE_BYTES
#undef SMEM_SMEM_N_STRIDE
#undef SMEM_SMEM_QD_OFF
#undef SMEM_SMEM_QD_STAGE_BYTES
#undef SMEM_SMEM_QD_STRIDE
#undef SMEM_SMEM_QD_TRANS_OFF
#undef SMEM_SMEM_QD_TRANS_STAGE_BYTES
#undef SMEM_SMEM_QD_TRANS_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_NCOLS
#undef TMEM_TMEM_DE_INP_OFFSET
#undef TMEM_TMEM_DH_INP_OFFSET
#undef TMEM_TMEM_DH_OFFSET
#undef TMEM_TMEM_DO_FINAL_OFFSET
#undef TMEM_TMEM_DO_INITIAL_OFFSET
#undef TMEM_TMEM_DR_INP_OFFSET
#undef TMEM_TMEM_DR_OFFSET
#undef TMEM_TMEM_DX_OFFSET
#undef de_inp_ready_addr
#undef dh_ready_addr
#undef dr_inp_ready_addr
#undef dr_ready_addr
#undef dx_ready_addr
#undef owners_done_addr
#undef prep_ready_addr
#undef smem_free_addr
#undef smem_j_addr
#undef smem_j_trans_addr
#undef smem_kd_addr
#undef smem_kd_trans_addr
#undef smem_ki_addr
#undef smem_kr_addr
#undef smem_n_addr
#undef smem_qd_addr
#undef smem_qd_trans_addr

#define FLASHKDA_INF CUDART_INF_F
#define TMEM_NCOLS 496
#define TMEM_TMEM_A_DH_OFFSET 0
#define TMEM_TMEM_A_STATE_OFFSET 64
#define TMEM_TMEM_A_K_OFFSET 128
#define TMEM_TMEM_A_Q_OFFSET 144
#define TMEM_TMEM_A_I_OFFSET 160
#define TMEM_TMEM_DKR_OFFSET 176
#define TMEM_TMEM_DI_OFFSET 208
#define TMEM_TMEM_DQ_LOCAL_OFFSET 240
#define TMEM_TMEM_DQ_BOUNDARY_OFFSET 272
#define TMEM_TMEM_DK_LOCAL_OFFSET 304
#define TMEM_TMEM_DK_BOUNDARY_OFFSET 336
#define TMEM_TMEM_RECON_KR_OFFSET 64
#define TMEM_TMEM_RECON_STATE_OFFSET 368
#define NUM_MAIN_STAGES 1
#define SMEM_S_K_OFF 1024
#define SMEM_S_K_STAGE_BYTES 8192
#define SMEM_S_K_STRIDE 8192
#define SMEM_S_I_OFF 9216
#define SMEM_S_I_STAGE_BYTES 8192
#define SMEM_S_I_STRIDE 8192
#define SMEM_S_DOT_OFF 17408
#define SMEM_S_DOT_STAGE_BYTES 8192
#define SMEM_S_DOT_STRIDE 8192
#define SMEM_S_RT_OFF 25600
#define SMEM_S_RT_STAGE_BYTES 8192
#define SMEM_S_RT_STRIDE 8192
#define SMEM_S_DRT_OFF 33792
#define SMEM_S_DRT_STAGE_BYTES 8192
#define SMEM_S_DRT_STRIDE 8192
#define SMEM_S_X_OFF 41984
#define SMEM_S_X_STAGE_BYTES 8192
#define SMEM_S_X_STRIDE 8192
#define SMEM_S_DET_OFF 50176
#define SMEM_S_DET_STAGE_BYTES 8192
#define SMEM_S_DET_STRIDE 8192
#define SMEM_S_DOT_TC_OFF 58368
#define SMEM_S_DOT_TC_STAGE_BYTES 8192
#define SMEM_S_DOT_TC_STRIDE 8192
#define SMEM_S_RT_TC_OFF 66560
#define SMEM_S_RT_TC_STAGE_BYTES 8192
#define SMEM_S_RT_TC_STRIDE 8192
#define SMEM_S_DET_TC_OFF 74752
#define SMEM_S_DET_TC_STAGE_BYTES 8192
#define SMEM_S_DET_TC_STRIDE 8192
#define SMEM_S_J_OFF 82944
#define SMEM_S_J_STAGE_BYTES 2048
#define SMEM_S_J_STRIDE 2048
#define SMEM_S_M_OFF 84992
#define SMEM_S_M_STAGE_BYTES 2048
#define SMEM_S_M_STRIDE 2048
#define SMEM_S_DJ_OFF 89088
#define SMEM_S_DJ_STAGE_BYTES 2048
#define SMEM_S_DJ_STRIDE 2048
#define SMEM_S_TMP_OFF 91136
#define SMEM_S_TMP_STAGE_BYTES 2048
#define SMEM_S_TMP_STRIDE 2048
#define SMEM_S_DF_OFF 93184
#define SMEM_S_DF_STAGE_BYTES 2048
#define SMEM_S_DF_STRIDE 2048
#define SMEM_S_DO_STAGE_OFF 89088
#define SMEM_S_DO_STAGE_STAGE_BYTES 8192
#define SMEM_S_DO_STAGE_STRIDE 8192
#define SMEM_S_DN_TC_OFF 97280
#define SMEM_S_DN_TC_STAGE_BYTES 2048
#define SMEM_S_DN_TC_STRIDE 2048
#define SMEM_S_DNT_TC_OFF 99328
#define SMEM_S_DNT_TC_STAGE_BYTES 2048
#define SMEM_S_DNT_TC_STRIDE 2048
#define SMEM_S_DM_TC_OFF 101376
#define SMEM_S_DM_TC_STAGE_BYTES 2048
#define SMEM_S_DM_TC_STRIDE 2048
#define SMEM_S_DMT_TC_OFF 103424
#define SMEM_S_DMT_TC_STAGE_BYTES 2048
#define SMEM_S_DMT_TC_STRIDE 2048
#define SMEM_S_DBETA_PARTIAL_OFF 105472
#define SMEM_S_DBETA_PARTIAL_STAGE_BYTES 512
#define SMEM_S_DBETA_PARTIAL_STRIDE 512
#define SMEM_S_Q_OFF 105984
#define SMEM_S_Q_STAGE_BYTES 8192
#define SMEM_S_Q_STRIDE 8192
#define SMEM_S_PREV_R_TC_OFF 114176
#define SMEM_S_PREV_R_TC_STAGE_BYTES 8192
#define SMEM_S_PREV_R_TC_STRIDE 8192
#define SMEM_S_DH_STAGE_OFF 122368
#define SMEM_S_DH_STAGE_STAGE_BYTES 32768
#define SMEM_S_DH_STAGE_STRIDE 32768
#define SMEM_S_STABLE_DKR_OFF 17408
#define SMEM_S_STABLE_DKR_STAGE_BYTES 16384
#define SMEM_S_STABLE_DKR_STRIDE 16384
#define SMEM_S_STABLE_DELTA_OFF 33792
#define SMEM_S_STABLE_DELTA_STAGE_BYTES 16384
#define SMEM_S_STABLE_DELTA_STRIDE 16384
#define SMEM_S_STABLE_BASE_OFF 50176
#define SMEM_S_STABLE_BASE_STAGE_BYTES 16384
#define SMEM_S_STABLE_BASE_STRIDE 16384
#define SMEM_S_MIDDLE_BASE_SUFFIX_OFF 122368
#define SMEM_S_MIDDLE_BASE_SUFFIX_STAGE_BYTES 512
#define SMEM_S_MIDDLE_BASE_SUFFIX_STRIDE 512
#define SMEM_S_HIGH_BASE_SUFFIX_OFF 122880
#define SMEM_S_HIGH_BASE_SUFFIX_STAGE_BYTES 512
#define SMEM_S_HIGH_BASE_SUFFIX_STRIDE 512
#define SMEM_TOTAL 155136
#define THREADS 384

extern "C" {

__global__ __launch_bounds__(384) void
kernel_flashkda_backward_local_c32_tcgen(__nv_bfloat16* __restrict__ do_, float* __restrict__ beta_active, long long* __restrict__ cu_seqlens, int* __restrict__ consumer_chunk_order, int* __restrict__ chunk_sequence, int* __restrict__ chunk_index, __nv_bfloat16* __restrict__ chunk_state, unsigned int* __restrict__ state_checkpoint_needed, float* __restrict__ initial_state, __nv_bfloat16* __restrict__ tape_qd, __nv_bfloat16* __restrict__ tape_kd, __nv_bfloat16* __restrict__ tape_kr, __nv_bfloat16* __restrict__ tape_j, float* __restrict__ tape_restore_factor, __nv_bfloat16* __restrict__ tape_e, __nv_bfloat16* __restrict__ tape_x, __nv_bfloat16* __restrict__ tape_r, __nv_bfloat16* __restrict__ chunk_dh, __nv_bfloat16* __restrict__ chunk_dr, __nv_bfloat16* __restrict__ chunk_dx, unsigned int* __restrict__ boundary_ready, __nv_bfloat16* __restrict__ grad_qd, __nv_bfloat16* __restrict__ grad_kd, __nv_bfloat16* __restrict__ grad_ki, float* __restrict__ dlog_decay, float* __restrict__ dbeta_active, __nv_bfloat16* __restrict__ dv, int num_heads, int boundary_ready_target, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* s_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int s_k_addr = smem + 1024;
    __nv_bfloat16* s_i = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int s_i_addr = smem + 9216;
    __nv_bfloat16* s_dot = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int s_dot_addr = smem + 17408;
    __nv_bfloat16* s_rt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int s_rt_addr = smem + 25600;
    __nv_bfloat16* s_drt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int s_drt_addr = smem + 33792;
    __nv_bfloat16* s_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int s_x_addr = smem + 41984;
    __nv_bfloat16* s_det = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int s_det_addr = smem + 50176;
    __nv_bfloat16* s_dot_tc = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int s_dot_tc_addr = smem + 58368;
    __nv_bfloat16* s_rt_tc = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int s_rt_tc_addr = smem + 66560;
    __nv_bfloat16* s_det_tc = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int s_det_tc_addr = smem + 74752;
    __nv_bfloat16* s_j = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int s_j_addr = smem + 82944;
    __nv_bfloat16* s_m = reinterpret_cast<__nv_bfloat16*>(smem_raw + 84992);
    const int s_m_addr = smem + 84992;
    __nv_bfloat16* s_dj = reinterpret_cast<__nv_bfloat16*>(smem_raw + 89088);
    const int s_dj_addr = smem + 89088;
    __nv_bfloat16* s_tmp = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
    const int s_tmp_addr = smem + 91136;
    __nv_bfloat16* s_df = reinterpret_cast<__nv_bfloat16*>(smem_raw + 93184);
    const int s_df_addr = smem + 93184;
    __nv_bfloat16* s_do_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 89088);
    const int s_do_stage_addr = smem + 89088;
    __nv_bfloat16* s_dn_tc = reinterpret_cast<__nv_bfloat16*>(smem_raw + 97280);
    const int s_dn_tc_addr = smem + 97280;
    __nv_bfloat16* s_dnt_tc = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int s_dnt_tc_addr = smem + 99328;
    __nv_bfloat16* s_dm_tc = reinterpret_cast<__nv_bfloat16*>(smem_raw + 101376);
    const int s_dm_tc_addr = smem + 101376;
    __nv_bfloat16* s_dmt_tc = reinterpret_cast<__nv_bfloat16*>(smem_raw + 103424);
    const int s_dmt_tc_addr = smem + 103424;
    float* s_dbeta_partial = reinterpret_cast<float*>(smem_raw + 105472);
    const int s_dbeta_partial_addr = smem + 105472;
    __nv_bfloat16* s_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 105984);
    const int s_q_addr = smem + 105984;
    __nv_bfloat16* s_prev_r_tc = reinterpret_cast<__nv_bfloat16*>(smem_raw + 114176);
    const int s_prev_r_tc_addr = smem + 114176;
    __nv_bfloat16* s_dh_stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 122368);
    const int s_dh_stage_addr = smem + 122368;
    float* s_stable_dkr = reinterpret_cast<float*>(smem_raw + 17408);
    const int s_stable_dkr_addr = smem + 17408;
    float* s_stable_delta = reinterpret_cast<float*>(smem_raw + 33792);
    const int s_stable_delta_addr = smem + 33792;
    float* s_stable_base = reinterpret_cast<float*>(smem_raw + 50176);
    const int s_stable_base_addr = smem + 50176;
    float* s_middle_base_suffix = reinterpret_cast<float*>(smem_raw + 122368);
    const int s_middle_base_suffix_addr = smem + 122368;
    float* s_high_base_suffix = reinterpret_cast<float*>(smem_raw + 122880);
    const int s_high_base_suffix_addr = smem + 122880;

    // Mbarrier init (15 groups, 15 barriers)
    // Mbarriers at smem_raw[0..120)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // prep_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // boundary_local_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // dbeta_done: 1 barriers, init_count=4
            mbarrier_init(smem + 16, 4);
            // qki_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // qki_tc_ready: 1 barriers, init_count=4
            mbarrier_init(smem + 32, 4);
            // dv_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // value_tc_ready: 1 barriers, init_count=3
            mbarrier_init(smem + 48, 3);
            // dh_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // state_tc_ready: 1 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            // recon_inputs_ready: 1 barriers, init_count=4
            mbarrier_init(smem + 72, 4);
            // recon_output_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // a_ready: 1 barriers, init_count=4
            mbarrier_init(smem + 88, 4);
            // first_outputs_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // outputs_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // epilogue_done: 1 barriers, init_count=4
            mbarrier_init(smem + 112, 4);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 496 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 120);
    if (warp == 0) {
        int _tmem_hold = smem + 120;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define prep_ready_addr (mbar_base + 0)
    #define boundary_local_ready_addr (mbar_base + 8)
    #define dbeta_done_addr (mbar_base + 16)
    #define qki_ready_addr (mbar_base + 24)
    #define qki_tc_ready_addr (mbar_base + 32)
    #define dv_ready_addr (mbar_base + 40)
    #define value_tc_ready_addr (mbar_base + 48)
    #define dh_ready_addr (mbar_base + 56)
    #define state_tc_ready_addr (mbar_base + 64)
    #define recon_inputs_ready_addr (mbar_base + 72)
    #define recon_output_ready_addr (mbar_base + 80)
    #define a_ready_addr (mbar_base + 88)
    #define first_outputs_ready_addr (mbar_base + 96)
    #define outputs_ready_addr (mbar_base + 104)
    #define epilogue_done_addr (mbar_base + 112)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_a_dh = taddr;
    const int tmem_tmem_a_state = taddr + 64;
    const int tmem_tmem_a_k = taddr + 128;
    const int tmem_tmem_a_q = taddr + 144;
    const int tmem_tmem_a_i = taddr + 160;
    const int tmem_tmem_dkr = taddr + 176;
    const int tmem_tmem_di = taddr + 208;
    const int tmem_tmem_dq_local = taddr + 240;
    const int tmem_tmem_dq_boundary = taddr + 272;
    const int tmem_tmem_dk_local = taddr + 304;
    const int tmem_tmem_dk_boundary = taddr + 336;
    const int tmem_tmem_recon_kr = taddr + 64;
    const int tmem_tmem_recon_state = taddr + 368;

    // ---- Role: prep ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // prep_main
            int ordered_chunk = blockIdx.x / num_heads;
            int chunk_global = consumer_chunk_order[ordered_chunk];
            int head = blockIdx.x - ordered_chunk * num_heads;
            int sequence = chunk_sequence[chunk_global];
            int local_chunk = chunk_index[chunk_global];
            long long bos = cu_seqlens[sequence];
            long long eos = cu_seqlens[sequence + 1];
            long long chunk_start = bos + (long long)local_chunk * 32;
            long long chunk_head = (long long)chunk_global * (long long)num_heads + (long long)head;
            int warp_id_in_role = (warp - 0);
            int prep_tid = warp_id_in_role * 32 + lane;
            long long token_key_base = chunk_head * 32 * 128;
            long long state_base = chunk_head * 128 * 128;
            long long restore_base = chunk_head * 128;
            int restore_key_col = prep_tid * 2 % 128;
            float _vec_load_0[2];
            {
                float2 _v2_0 = *reinterpret_cast<const float2*>(tape_restore_factor + restore_base + (long long)restore_key_col);
                _vec_load_0[0] = _v2_0.x;
                _vec_load_0[0 + 1] = _v2_0.y;
            }
            float _rcp_0 = approx_rcp(_vec_load_0[0]);
            float inv_restore0 = _rcp_0;
            float _rcp_1 = approx_rcp(_vec_load_0[1]);
            float inv_restore1 = _rcp_1;
            #pragma unroll
            for (int key_copy_pass = 0; key_copy_pass < 4; key_copy_pass++) {
                int key_copy_item = key_copy_pass * 128 * 8 + prep_tid * 8;
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(s_k_addr + (unsigned int)((key_copy_item * 2 ^ (key_copy_item * 2 >> 8 & 7) << 4) / 2 * 2)), "l"(tape_kd + (token_key_base + (long long)key_copy_item)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(s_q_addr + (unsigned int)((key_copy_item * 2 ^ (key_copy_item * 2 >> 8 & 7) << 4) / 2 * 2)), "l"(tape_qd + (token_key_base + (long long)key_copy_item)));
            }
            asm volatile("cp.async.commit_group;");
            #pragma unroll
            for (int key_pass = 0; key_pass < 16; key_pass++) {
                int item = key_pass * 256 + prep_tid * 2;
                float _vec_load_1[2];
                {
                    uint32_t _bf16x2_bits_1;
                    _bf16x2_bits_1 = *reinterpret_cast<const uint32_t*>(tape_kr + token_key_base + (long long)item);
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_1[0])[0]), "=f"((&_vec_load_1[0])[1])
                        : "r"(_bf16x2_bits_1));
                }
                float i_values[2];
                unsigned int i_packed[1];
                i_values[0] = _vec_load_1[0] * inv_restore0;
                i_values[1] = _vec_load_1[1] * inv_restore1;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(i_values[_lp*2 + 0], i_values[_lp*2+1 + 0]));
                    i_packed[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_i_addr + (unsigned int)((item * 2 ^ (item * 2 >> 8 & 7) << 4) / 2 * 2)), "r"((i_packed[0])));
            }
            asm volatile("cp.async.wait_group 0;");
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp_id_in_role == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(qki_ready_addr);
                }
            }
            long long j_base = chunk_head * 32 * 32;
            #pragma unroll
            for (int j_pass = 0; j_pass < 8; j_pass++) {
                int item_1 = j_pass * 128 + prep_tid;
                s_j[(item_1 * 2 ^ (item_1 * 2 >> 7 & 7) << 4) / 2] = tape_j[j_base + (long long)item_1];
            }
            long long token_value_base = chunk_head * 128 * 32;
            #pragma unroll
            for (int value_copy_pass = 0; value_copy_pass < 4; value_copy_pass++) {
                int value_copy_item = value_copy_pass * 128 * 8 + prep_tid * 8;
                int value_copy_dst = (value_copy_item * 2 ^ (value_copy_item * 2 >> 7 & 7) << 4) / 2 * 2;
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(s_x_addr + (unsigned int)value_copy_dst), "l"(tape_x + (token_value_base + (long long)value_copy_item)));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(s_rt_addr + (unsigned int)value_copy_dst), "l"(tape_r + (token_value_base + (long long)value_copy_item)));
            }
            asm volatile("cp.async.commit_group;");
            int tile = warp_id_in_role;
            int row_base = tile / 2 * 16;
            int col_base = tile % 2 * 16;
            float acc[8];
            int lane_matrix = lane / 8;
            int lane_row8 = lane & 7;
            #pragma unroll
            for (int kk = 0; kk < 128; kk += 16) {
                unsigned int a_plain[4];
                unsigned int a_trans[4];
                unsigned int b_plain[4];
                unsigned int b_trans[4];
                {
                    int a_row = row_base + lane_row8 + (lane_matrix & 1) * 8;
                    int a_col = kk + lane_matrix / 2 * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_plain[0]), "=r"(a_plain[1]), "=r"(a_plain[2]), "=r"(a_plain[3])
                        : "r"(s_k_addr + (unsigned int)(((1) ? ((a_row * 128 + a_col) * 2 ^ ((a_row * 128 + a_col) * 2 >> 8 & 7) << 4) / 2 : ((a_row * 128 + a_col) * 2 ^ ((a_row * 128 + a_col) * 2 >> 7 & 7) << 4) / 2) * 2))
                        : "memory");
                }
                {
                    #pragma unroll
                    for (int n_half = 0; n_half < 2; n_half++) {
                        int b_row = col_base + n_half * 8 + lane_row8;
                        int b_col = kk + lane / 8 * 8;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_plain[n_half * 2]), "=r"(b_plain[n_half * 2 + 1])
                            : "r"(s_i_addr + (unsigned int)(((1) ? ((b_row * 128 + b_col) * 2 ^ ((b_row * 128 + b_col) * 2 >> 8 & 7) << 4) / 2 : ((b_row * 128 + b_col) * 2 ^ ((b_row * 128 + b_col) * 2 >> 7 & 7) << 4) / 2) * 2))
                            : "memory");
                    }
                }
                {
                    {
                        {
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                                : "r"(a_plain[0]), "r"(a_plain[1]), "r"(a_plain[2]), "r"(a_plain[3]), "r"(b_plain[0]), "r"(b_plain[1]), "f"(((kk == 0) ? 0.0f : acc[0])), "f"(((kk == 0) ? 0.0f : acc[1])), "f"(((kk == 0) ? 0.0f : acc[2])), "f"(((kk == 0) ? 0.0f : acc[3])));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                                : "r"(a_plain[0]), "r"(a_plain[1]), "r"(a_plain[2]), "r"(a_plain[3]), "r"(b_plain[2]), "r"(b_plain[(2) + 1]), "f"(((kk == 0) ? 0.0f : acc[4])), "f"(((kk == 0) ? 0.0f : acc[(4) + 1])), "f"(((kk == 0) ? 0.0f : acc[(4) + 2])), "f"(((kk == 0) ? 0.0f : acc[(4) + 3])));
                        }
                    }
                }
            }
            int frag_row = lane / 4;
            int frag_col = (lane & 3) * 2;
            #pragma unroll
            for (int n_half_1 = 0; n_half_1 < 2; n_half_1++) {
                int frag = n_half_1 * 4;
                int row0 = row_base + frag_row;
                int row1 = row0 + 8;
                int col0 = col_base + n_half_1 * 8 + frag_col;
                float pair[2];
                unsigned int packed[1];
                pair[0] = acc[frag];
                pair[1] = acc[frag + 1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_m_addr + (unsigned int)(((row0 * 32 + col0) * 2 ^ ((row0 * 32 + col0) * 2 >> 7 & 7) << 4) / 2 * 2)), "r"((packed[0])));
                pair[0] = acc[frag + 2];
                pair[1] = acc[frag + 3];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_m_addr + (unsigned int)(((row1 * 32 + col0) * 2 ^ ((row1 * 32 + col0) * 2 >> 7 & 7) << 4) / 2 * 2)), "r"((packed[0])));
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (boundary_ready_target != 0) {
                if (warp_id_in_role == 0) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(boundary_ready) + (chunk_head);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(boundary_ready_target)) break;
                        }
                    }
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (warp_id_in_role == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(boundary_local_ready_addr);
                }
            }
            #pragma unroll
            for (int dh_copy_pass = 0; dh_copy_pass < 16; dh_copy_pass++) {
                int dh_copy_item = dh_copy_pass * 128 * 8 + prep_tid * 8;
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(s_dh_stage_addr + (unsigned int)((dh_copy_item * 2 ^ (dh_copy_item * 2 >> 7 & 7) << 4) / 2 * 2)), "l"(chunk_dh + (state_base + (long long)dh_copy_item)));
            }
            asm volatile("cp.async.commit_group;");
            #pragma unroll
            for (int boundary_copy_pass = 0; boundary_copy_pass < 4; boundary_copy_pass++) {
                int boundary_copy_item = boundary_copy_pass * 128 * 8 + prep_tid * 8;
                int boundary_copy_dst = (boundary_copy_item * 2 ^ (boundary_copy_item * 2 >> 7 & 7) << 4) / 2 * 2;
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(s_dot_addr + (unsigned int)boundary_copy_dst), "l"(chunk_dx + (token_value_base + (long long)boundary_copy_item)));
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 2;");
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            int value_token = lane;
            long long value_token_global = chunk_start + (long long)value_token;
            float value_beta = 0.0f;
            if (value_token_global < eos) {
                value_beta = beta_active[value_token_global * (long long)num_heads + (long long)head];
            }
            #pragma unroll
            for (int value_segment = 0; value_segment < 4; value_segment++) {
                int tc_segment = warp_id_in_role * 4 + value_segment;
                float r_values[8];
                #pragma unroll
                for (int value_elem = 0; value_elem < 8; value_elem++) {
                    int value_row = tc_segment * 8 + value_elem;
                    int value_major_item = value_row * 32 + value_token;
                    __nv_bfloat16 r_value = s_rt[(value_major_item * 2 ^ (value_major_item * 2 >> 7 & 7) << 4) / 2];
                    float _cvt_f32_0 = __bfloat162float(r_value);
                    r_values[value_elem] = _cvt_f32_0;
                }
                unsigned int packed_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(r_values[_lp*2 + 0], r_values[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_rt_tc_addr + ((s_rt_tc_addr + (unsigned int)(tc_segment * 8 / 64 * 4096 + value_token * 128 + tc_segment * 8 % 64 * 2 ^ (tc_segment * 8 / 64 * 4096 + value_token * 128 + tc_segment * 8 % 64 * 2 >> 7 & 7) << 4)) - s_rt_tc_addr)), "r"(packed_1[0]), "r"(packed_1[1]), "r"(packed_1[2]), "r"(packed_1[3]) : "memory");
            }
            asm volatile("cp.async.wait_group 1;");
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp_id_in_role == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(dh_ready_addr);
                }
            }
            asm volatile("cp.async.wait_group 0;");
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            #pragma unroll
            for (int value_segment_1 = 0; value_segment_1 < 4; value_segment_1++) {
                int tc_segment_1 = warp_id_in_role * 4 + value_segment_1;
                float det_values[8];
                #pragma unroll
                for (int value_elem_1 = 0; value_elem_1 < 8; value_elem_1++) {
                    int value_row_1 = tc_segment_1 * 8 + value_elem_1;
                    int value_major_item_1 = value_row_1 * 32 + value_token;
                    __nv_bfloat16 dx_value = s_dot[(value_major_item_1 * 2 ^ (value_major_item_1 * 2 >> 7 & 7) << 4) / 2];
                    int det_group = tc_segment_1 * 2 + value_elem_1 / 4;
                    s_det[value_row_1 * 32 + (value_token ^ det_group)] = dx_value;
                    float _cvt_f32_1 = __bfloat162float(dx_value);
                    det_values[value_elem_1] = _cvt_f32_1 * value_beta;
                }
                unsigned int packed_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(det_values[_lp*2 + 0], det_values[_lp*2+1 + 0]));
                    packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_det_tc_addr + ((s_det_tc_addr + (unsigned int)(tc_segment_1 * 8 / 64 * 4096 + value_token * 128 + tc_segment_1 * 8 % 64 * 2 ^ (tc_segment_1 * 8 / 64 * 4096 + value_token * 128 + tc_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) - s_det_tc_addr)), "r"(packed_2[0]), "r"(packed_2[1]), "r"(packed_2[2]), "r"(packed_2[3]) : "memory");
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (warp_id_in_role == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(dv_ready_addr);
                }
            }
            unsigned int _phase_value_tc_ready_0 = 0;
            mbarrier_wait(value_tc_ready_addr, _phase_value_tc_ready_0);
            _phase_value_tc_ready_0 ^= 1;
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            int lane_matrix_0 = lane / 8;
            int lane_row8_1 = lane & 7;
            #pragma unroll
            for (int kk_1 = 0; kk_1 < 128; kk_1 += 16) {
                unsigned int a_plain_1[4];
                unsigned int a_trans_1[4];
                unsigned int b_plain_1[4];
                unsigned int b_trans_1[4];
                {
                    int a_row_1 = kk_1 + lane_row8_1 + lane_matrix_0 / 2 * 8;
                    int a_col_1 = row_base + (lane_matrix_0 & 1) * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_trans_1[0]), "=r"(a_trans_1[1]), "=r"(a_trans_1[2]), "=r"(a_trans_1[3])
                        : "r"(s_rt_addr + (unsigned int)(((0) ? ((a_row_1 * 32 + a_col_1) * 2 ^ ((a_row_1 * 32 + a_col_1) * 2 >> 8 & 7) << 4) / 2 : ((a_row_1 * 32 + a_col_1) * 2 ^ ((a_row_1 * 32 + a_col_1) * 2 >> 7 & 7) << 4) / 2) * 2))
                        : "memory");
                }
                {
                    #pragma unroll
                    for (int n_half_2 = 0; n_half_2 < 2; n_half_2++) {
                        int b_row_1 = col_base + n_half_2 * 8 + lane_row8_1;
                        int b_col_1 = kk_1 + lane / 8 * 8;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_plain_1[n_half_2 * 2]), "=r"(b_plain_1[n_half_2 * 2 + 1])
                            : "r"(s_do_stage_addr + (unsigned int)(((1) ? ((b_row_1 * 128 + b_col_1) * 2 ^ ((b_row_1 * 128 + b_col_1) * 2 >> 8 & 7) << 4) / 2 : ((b_row_1 * 128 + b_col_1) * 2 ^ ((b_row_1 * 128 + b_col_1) * 2 >> 7 & 7) << 4) / 2) * 2))
                            : "memory");
                    }
                }
                {
                    {
                        {
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                                : "r"(a_trans_1[0]), "r"(a_trans_1[1]), "r"(a_trans_1[2]), "r"(a_trans_1[3]), "r"(b_plain_1[0]), "r"(b_plain_1[1]), "f"(((kk_1 == 0) ? 0.0f : acc[0])), "f"(((kk_1 == 0) ? 0.0f : acc[1])), "f"(((kk_1 == 0) ? 0.0f : acc[2])), "f"(((kk_1 == 0) ? 0.0f : acc[3])));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                                : "r"(a_trans_1[0]), "r"(a_trans_1[1]), "r"(a_trans_1[2]), "r"(a_trans_1[3]), "r"(b_plain_1[2]), "r"(b_plain_1[(2) + 1]), "f"(((kk_1 == 0) ? 0.0f : acc[4])), "f"(((kk_1 == 0) ? 0.0f : acc[(4) + 1])), "f"(((kk_1 == 0) ? 0.0f : acc[(4) + 2])), "f"(((kk_1 == 0) ? 0.0f : acc[(4) + 3])));
                        }
                    }
                }
            }
            float dn_values[8];
            dn_values[0] = 0.0f;
            dn_values[1] = 0.0f;
            dn_values[2] = 0.0f;
            dn_values[3] = 0.0f;
            dn_values[4] = 0.0f;
            dn_values[5] = 0.0f;
            dn_values[6] = 0.0f;
            dn_values[7] = 0.0f;
            int dn_row0 = row_base + lane / 4;
            int dn_row1 = dn_row0 + 8;
            int dn_col0 = col_base + (lane & 3) * 2;
            if (dn_row0 <= dn_col0) {
                dn_values[0] = acc[0];
            }
            if (dn_row0 <= dn_col0 + 1) {
                dn_values[1] = acc[1];
            }
            if (dn_row1 <= dn_col0) {
                dn_values[2] = acc[2];
            }
            if (dn_row1 <= dn_col0 + 1) {
                dn_values[3] = acc[3];
            }
            if (dn_row0 <= dn_col0 + 8) {
                dn_values[4] = acc[4];
            }
            if (dn_row0 <= dn_col0 + 9) {
                dn_values[5] = acc[5];
            }
            if (dn_row1 <= dn_col0 + 8) {
                dn_values[6] = acc[6];
            }
            if (dn_row1 <= dn_col0 + 9) {
                dn_values[7] = acc[7];
            }
            unsigned int packed_3[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dn_values[_lp*2 + 0], dn_values[_lp*2+1 + 0]));
                packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            int direct_row = row_base + lane % 16;
            int direct_col = col_base + lane / 16 * 8;
            uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)(s_dnt_tc_addr + (unsigned int)(direct_col / 16 * 1024 + direct_row * 32 + direct_col % 16 * 2 ^ (direct_col / 16 * 1024 + direct_row * 32 + direct_col % 16 * 2 >> 7 & 1) << 4)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[3]))
                : "memory");
            #pragma unroll
            for (int publish_pair = 0; publish_pair < 2; publish_pair++) {
                int transpose_row = col_base + publish_pair * 8 + (lane & 7);
                int transpose_col = row_base + lane / 8 * 8;
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>((unsigned long long)(s_dn_tc_addr + (unsigned int)(transpose_col / 16 * 1024 + transpose_row * 32 + transpose_col % 16 * 2 ^ (transpose_col / 16 * 1024 + transpose_row * 32 + transpose_col % 16 * 2 >> 7 & 1) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[publish_pair * 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[publish_pair * 2 + 1]))
                    : "memory");
            }
            int lane_matrix_2 = lane / 8;
            int lane_row8_3 = lane & 7;
            #pragma unroll
            for (int kk_2 = 0; kk_2 < 128; kk_2 += 16) {
                unsigned int a_plain_2[4];
                unsigned int a_trans_2[4];
                unsigned int b_plain_2[4];
                unsigned int b_trans_2[4];
                {
                    int a_row_2 = kk_2 + lane_row8_3 + lane_matrix_2 / 2 * 8;
                    int a_col_2 = row_base + (lane_matrix_2 & 1) * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_trans_2[0]), "=r"(a_trans_2[1]), "=r"(a_trans_2[2]), "=r"(a_trans_2[3])
                        : "r"(s_drt_addr + (unsigned int)(((0) ? ((a_row_2 * 32 + a_col_2) * 2 ^ ((a_row_2 * 32 + a_col_2) * 2 >> 8 & 7) << 4) / 2 : ((a_row_2 * 32 + a_col_2) * 2 ^ ((a_row_2 * 32 + a_col_2) * 2 >> 7 & 7) << 4) / 2) * 2))
                        : "memory");
                }
                {
                    #pragma unroll
                    for (int n_half_3 = 0; n_half_3 < 2; n_half_3++) {
                        int b_row_2 = kk_2 + lane % 16;
                        int b_col_2 = col_base + n_half_3 * 8;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_trans_2[n_half_3 * 2]), "=r"(b_trans_2[n_half_3 * 2 + 1])
                            : "r"(s_x_addr + (unsigned int)(((0) ? ((b_row_2 * 32 + b_col_2) * 2 ^ ((b_row_2 * 32 + b_col_2) * 2 >> 8 & 7) << 4) / 2 : ((b_row_2 * 32 + b_col_2) * 2 ^ ((b_row_2 * 32 + b_col_2) * 2 >> 7 & 7) << 4) / 2) * 2))
                            : "memory");
                    }
                }
                {
                    {
                        {
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                                : "r"(a_trans_2[0]), "r"(a_trans_2[1]), "r"(a_trans_2[2]), "r"(a_trans_2[3]), "r"(b_trans_2[0]), "r"(b_trans_2[1]), "f"(((kk_2 == 0) ? 0.0f : acc[0])), "f"(((kk_2 == 0) ? 0.0f : acc[1])), "f"(((kk_2 == 0) ? 0.0f : acc[2])), "f"(((kk_2 == 0) ? 0.0f : acc[3])));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                                : "r"(a_trans_2[0]), "r"(a_trans_2[1]), "r"(a_trans_2[2]), "r"(a_trans_2[3]), "r"(b_trans_2[2]), "r"(b_trans_2[(2) + 1]), "f"(((kk_2 == 0) ? 0.0f : acc[4])), "f"(((kk_2 == 0) ? 0.0f : acc[(4) + 1])), "f"(((kk_2 == 0) ? 0.0f : acc[(4) + 2])), "f"(((kk_2 == 0) ? 0.0f : acc[(4) + 3])));
                        }
                    }
                }
            }
            int frag_row_4 = lane / 4;
            int frag_col_5 = (lane & 3) * 2;
            #pragma unroll
            for (int n_half_4 = 0; n_half_4 < 2; n_half_4++) {
                int frag_1 = n_half_4 * 4;
                int row0_1 = row_base + frag_row_4;
                int row1_1 = row0_1 + 8;
                int col0_1 = col_base + n_half_4 * 8 + frag_col_5;
                float pair_1[2];
                unsigned int packed_0[1];
                pair_1[0] = acc[frag_1];
                pair_1[1] = acc[frag_1 + 1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_1[_lp*2 + 0], pair_1[_lp*2+1 + 0]));
                    packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_dj_addr + (unsigned int)(((row0_1 * 32 + col0_1) * 2 ^ ((row0_1 * 32 + col0_1) * 2 >> 7 & 7) << 4) / 2 * 2)), "r"((packed_0[0])));
                pair_1[0] = acc[frag_1 + 2];
                pair_1[1] = acc[frag_1 + 3];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_1[_lp*2 + 0], pair_1[_lp*2+1 + 0]));
                    packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_dj_addr + (unsigned int)(((row1_1 * 32 + col0_1) * 2 ^ ((row1_1 * 32 + col0_1) * 2 >> 7 & 7) << 4) / 2 * 2)), "r"((packed_0[0])));
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            #pragma unroll
            for (int mask_pass = 0; mask_pass < 8; mask_pass++) {
                int item_2 = mask_pass * 128 + prep_tid;
                int row = item_2 / 32;
                int col = item_2 % 32;
                if (row <= col) {
                    s_m[(item_2 * 2 ^ (item_2 * 2 >> 7 & 7) << 4) / 2] = 0.0f;
                }
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            int lane_matrix_6 = lane / 8;
            int lane_row8_7 = lane & 7;
            #pragma unroll
            for (int kk_3 = 0; kk_3 < 32; kk_3 += 16) {
                unsigned int a_plain_3[4];
                unsigned int a_trans_3[4];
                unsigned int b_plain_3[4];
                unsigned int b_trans_3[4];
                {
                    int a_row_3 = kk_3 + lane_row8_7 + lane_matrix_6 / 2 * 8;
                    int a_col_3 = row_base + (lane_matrix_6 & 1) * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_trans_3[0]), "=r"(a_trans_3[1]), "=r"(a_trans_3[2]), "=r"(a_trans_3[3])
                        : "r"(s_j_addr + (unsigned int)(((0) ? ((a_row_3 * 32 + a_col_3) * 2 ^ ((a_row_3 * 32 + a_col_3) * 2 >> 8 & 7) << 4) / 2 : ((a_row_3 * 32 + a_col_3) * 2 ^ ((a_row_3 * 32 + a_col_3) * 2 >> 7 & 7) << 4) / 2) * 2))
                        : "memory");
                }
                {
                    #pragma unroll
                    for (int n_half_5 = 0; n_half_5 < 2; n_half_5++) {
                        int b_row_3 = kk_3 + lane % 16;
                        int b_col_3 = col_base + n_half_5 * 8;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_trans_3[n_half_5 * 2]), "=r"(b_trans_3[n_half_5 * 2 + 1])
                            : "r"(s_dj_addr + (unsigned int)(((0) ? ((b_row_3 * 32 + b_col_3) * 2 ^ ((b_row_3 * 32 + b_col_3) * 2 >> 8 & 7) << 4) / 2 : ((b_row_3 * 32 + b_col_3) * 2 ^ ((b_row_3 * 32 + b_col_3) * 2 >> 7 & 7) << 4) / 2) * 2))
                            : "memory");
                    }
                }
                {
                    {
                        {
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                                : "r"(a_trans_3[0]), "r"(a_trans_3[1]), "r"(a_trans_3[2]), "r"(a_trans_3[3]), "r"(b_trans_3[0]), "r"(b_trans_3[1]), "f"(((kk_3 == 0) ? 0.0f : acc[0])), "f"(((kk_3 == 0) ? 0.0f : acc[1])), "f"(((kk_3 == 0) ? 0.0f : acc[2])), "f"(((kk_3 == 0) ? 0.0f : acc[3])));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                                : "r"(a_trans_3[0]), "r"(a_trans_3[1]), "r"(a_trans_3[2]), "r"(a_trans_3[3]), "r"(b_trans_3[2]), "r"(b_trans_3[(2) + 1]), "f"(((kk_3 == 0) ? 0.0f : acc[4])), "f"(((kk_3 == 0) ? 0.0f : acc[(4) + 1])), "f"(((kk_3 == 0) ? 0.0f : acc[(4) + 2])), "f"(((kk_3 == 0) ? 0.0f : acc[(4) + 3])));
                        }
                    }
                }
            }
            int frag_row_8 = lane / 4;
            int frag_col_9 = (lane & 3) * 2;
            #pragma unroll
            for (int n_half_6 = 0; n_half_6 < 2; n_half_6++) {
                int frag_2 = n_half_6 * 4;
                int row0_2 = row_base + frag_row_8;
                int row1_2 = row0_2 + 8;
                int col0_2 = col_base + n_half_6 * 8 + frag_col_9;
                float pair_2[2];
                unsigned int packed_0_1[1];
                pair_2[0] = acc[frag_2];
                pair_2[1] = acc[frag_2 + 1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_2[_lp*2 + 0], pair_2[_lp*2+1 + 0]));
                    packed_0_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_tmp_addr + (unsigned int)(((row0_2 * 32 + col0_2) * 2 ^ ((row0_2 * 32 + col0_2) * 2 >> 7 & 7) << 4) / 2 * 2)), "r"((packed_0_1[0])));
                pair_2[0] = acc[frag_2 + 2];
                pair_2[1] = acc[frag_2 + 3];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_2[_lp*2 + 0], pair_2[_lp*2+1 + 0]));
                    packed_0_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_tmp_addr + (unsigned int)(((row1_2 * 32 + col0_2) * 2 ^ ((row1_2 * 32 + col0_2) * 2 >> 7 & 7) << 4) / 2 * 2)), "r"((packed_0_1[0])));
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            int lane_matrix_10 = lane / 8;
            int lane_row8_11 = lane & 7;
            #pragma unroll
            for (int kk_4 = 0; kk_4 < 32; kk_4 += 16) {
                unsigned int a_plain_4[4];
                unsigned int a_trans_4[4];
                unsigned int b_plain_4[4];
                unsigned int b_trans_4[4];
                {
                    int a_row_4 = row_base + lane_row8_11 + (lane_matrix_10 & 1) * 8;
                    int a_col_4 = kk_4 + lane_matrix_10 / 2 * 8;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_plain_4[0]), "=r"(a_plain_4[1]), "=r"(a_plain_4[2]), "=r"(a_plain_4[3])
                        : "r"(s_tmp_addr + (unsigned int)(((0) ? ((a_row_4 * 32 + a_col_4) * 2 ^ ((a_row_4 * 32 + a_col_4) * 2 >> 8 & 7) << 4) / 2 : ((a_row_4 * 32 + a_col_4) * 2 ^ ((a_row_4 * 32 + a_col_4) * 2 >> 7 & 7) << 4) / 2) * 2))
                        : "memory");
                }
                {
                    #pragma unroll
                    for (int n_half_7 = 0; n_half_7 < 2; n_half_7++) {
                        int b_row_4 = col_base + n_half_7 * 8 + lane_row8_11;
                        int b_col_4 = kk_4 + lane / 8 * 8;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_plain_4[n_half_7 * 2]), "=r"(b_plain_4[n_half_7 * 2 + 1])
                            : "r"(s_j_addr + (unsigned int)(((0) ? ((b_row_4 * 32 + b_col_4) * 2 ^ ((b_row_4 * 32 + b_col_4) * 2 >> 8 & 7) << 4) / 2 : ((b_row_4 * 32 + b_col_4) * 2 ^ ((b_row_4 * 32 + b_col_4) * 2 >> 7 & 7) << 4) / 2) * 2))
                            : "memory");
                    }
                }
                {
                    {
                        {
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                                : "r"(a_plain_4[0]), "r"(a_plain_4[1]), "r"(a_plain_4[2]), "r"(a_plain_4[3]), "r"(b_plain_4[0]), "r"(b_plain_4[1]), "f"(((kk_4 == 0) ? 0.0f : acc[0])), "f"(((kk_4 == 0) ? 0.0f : acc[1])), "f"(((kk_4 == 0) ? 0.0f : acc[2])), "f"(((kk_4 == 0) ? 0.0f : acc[3])));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                                : "r"(a_plain_4[0]), "r"(a_plain_4[1]), "r"(a_plain_4[2]), "r"(a_plain_4[3]), "r"(b_plain_4[2]), "r"(b_plain_4[(2) + 1]), "f"(((kk_4 == 0) ? 0.0f : acc[4])), "f"(((kk_4 == 0) ? 0.0f : acc[(4) + 1])), "f"(((kk_4 == 0) ? 0.0f : acc[(4) + 2])), "f"(((kk_4 == 0) ? 0.0f : acc[(4) + 3])));
                        }
                    }
                }
            }
            const float2 _scale2_4 = {-1.0f, -1.0f};
            #pragma unroll
            for (int _ls = 0; _ls < 4; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(acc)[_ls], _scale2_4);
            int frag_row_12 = lane / 4;
            int frag_col_13 = (lane & 3) * 2;
            #pragma unroll
            for (int n_half_8 = 0; n_half_8 < 2; n_half_8++) {
                int frag_3 = n_half_8 * 4;
                int row0_3 = row_base + frag_row_12;
                int row1_3 = row0_3 + 8;
                int col0_3 = col_base + n_half_8 * 8 + frag_col_13;
                float pair_3[2];
                unsigned int packed_0_2[1];
                pair_3[0] = acc[frag_3];
                pair_3[1] = acc[frag_3 + 1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_3[_lp*2 + 0], pair_3[_lp*2+1 + 0]));
                    packed_0_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_df_addr + (unsigned int)(((row0_3 * 32 + col0_3) * 2 ^ ((row0_3 * 32 + col0_3) * 2 >> 7 & 7) << 4) / 2 * 2)), "r"((packed_0_2[0])));
                pair_3[0] = acc[frag_3 + 2];
                pair_3[1] = acc[frag_3 + 3];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_3[_lp*2 + 0], pair_3[_lp*2+1 + 0]));
                    packed_0_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(s_df_addr + (unsigned int)(((row1_3 * 32 + col0_3) * 2 ^ ((row1_3 * 32 + col0_3) * 2 >> 7 & 7) << 4) / 2 * 2)), "r"((packed_0_2[0])));
            }
            float _shfl_0 = __shfl_sync(0xFFFFFFFF, value_beta, dn_row0);
            float dm_beta0 = _shfl_0;
            float _shfl_1 = __shfl_sync(0xFFFFFFFF, value_beta, dn_row1);
            float dm_beta1 = _shfl_1;
            float dm_values[8];
            dm_values[0] = 0.0f;
            dm_values[1] = 0.0f;
            dm_values[2] = 0.0f;
            dm_values[3] = 0.0f;
            dm_values[4] = 0.0f;
            dm_values[5] = 0.0f;
            dm_values[6] = 0.0f;
            dm_values[7] = 0.0f;
            if (dn_row0 > dn_col0) {
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(acc[0]);
                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_0);
                dm_values[0] = _cvt_f32_2 * dm_beta0;
            }
            if (dn_row0 > dn_col0 + 1) {
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(acc[1]);
                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_1);
                dm_values[1] = _cvt_f32_3 * dm_beta0;
            }
            if (dn_row1 > dn_col0) {
                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(acc[2]);
                float _cvt_f32_4 = __bfloat162float(_cvt_bf16_2);
                dm_values[2] = _cvt_f32_4 * dm_beta1;
            }
            if (dn_row1 > dn_col0 + 1) {
                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(acc[3]);
                float _cvt_f32_5 = __bfloat162float(_cvt_bf16_3);
                dm_values[3] = _cvt_f32_5 * dm_beta1;
            }
            if (dn_row0 > dn_col0 + 8) {
                __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(acc[4]);
                float _cvt_f32_6 = __bfloat162float(_cvt_bf16_4);
                dm_values[4] = _cvt_f32_6 * dm_beta0;
            }
            if (dn_row0 > dn_col0 + 9) {
                __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(acc[5]);
                float _cvt_f32_7 = __bfloat162float(_cvt_bf16_5);
                dm_values[5] = _cvt_f32_7 * dm_beta0;
            }
            if (dn_row1 > dn_col0 + 8) {
                __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(acc[6]);
                float _cvt_f32_8 = __bfloat162float(_cvt_bf16_6);
                dm_values[6] = _cvt_f32_8 * dm_beta1;
            }
            if (dn_row1 > dn_col0 + 9) {
                __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(acc[7]);
                float _cvt_f32_9 = __bfloat162float(_cvt_bf16_7);
                dm_values[7] = _cvt_f32_9 * dm_beta1;
            }
            unsigned int packed_14[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dm_values[_lp*2 + 0], dm_values[_lp*2+1 + 0]));
                packed_14[_lp] = *(uint32_t*)&_bf2;
            }
            int direct_row_15 = row_base + lane % 16;
            int direct_col_16 = col_base + lane / 16 * 8;
            uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)(s_dmt_tc_addr + (unsigned int)(direct_col_16 / 16 * 1024 + direct_row_15 * 32 + direct_col_16 % 16 * 2 ^ (direct_col_16 / 16 * 1024 + direct_row_15 * 32 + direct_col_16 % 16 * 2 >> 7 & 1) << 4)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[3]))
                : "memory");
            #pragma unroll
            for (int publish_pair_1 = 0; publish_pair_1 < 2; publish_pair_1++) {
                int transpose_row_1 = col_base + publish_pair_1 * 8 + (lane & 7);
                int transpose_col_1 = row_base + lane / 8 * 8;
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>((unsigned long long)(s_dm_tc_addr + (unsigned int)(transpose_col_1 / 16 * 1024 + transpose_row_1 * 32 + transpose_col_1 % 16 * 2 ^ (transpose_col_1 / 16 * 1024 + transpose_row_1 * 32 + transpose_col_1 % 16 * 2 >> 7 & 1) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[publish_pair_1 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_14[publish_pair_1 * 2 + 1]))
                    : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp_id_in_role == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(prep_ready_addr);
                }
            }
            int token_col = lane;
            float db_partial = 0.0f;
            float inv_value_beta = 0.0f;
            if (value_beta != 0.0f) {
                float _rcp_2 = approx_rcp(value_beta);
                inv_value_beta = _rcp_2;
            }
            #pragma unroll
            for (int value_local = 0; value_local < 32; value_local++) {
                int value_row_2 = warp_id_in_role * 32 + value_local;
                long long tape_item = token_value_base + (long long)(value_row_2 * 32) + (long long)token_col;
                int value_major_item_2 = value_row_2 * 32 + token_col;
                __nv_bfloat16 x_value = s_x[(value_major_item_2 * 2 ^ (value_major_item_2 * 2 >> 7 & 7) << 4) / 2];
                float _cvt_f32_10 = __bfloat162float(x_value);
                float recovered_e = _cvt_f32_10 * inv_value_beta;
                float _fma_0 = __fmaf_rn((float)chunk_dx[tape_item], recovered_e, db_partial);
                db_partial = _fma_0;
            }
            int db_matrix_item = token_col * 32 + warp_id_in_role * 8;
            unsigned int db_df_packed[4];
            unsigned int db_m_packed[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&db_df_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&db_df_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&db_df_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&db_df_packed[(0) + 3]))
                : "r"(s_df_addr + (unsigned int)((db_matrix_item * 2 ^ (db_matrix_item * 2 >> 7 & 7) << 4) / 2 * 2)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&db_m_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&db_m_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&db_m_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&db_m_packed[(0) + 3]))
                : "r"(s_m_addr + (unsigned int)((db_matrix_item * 2 ^ (db_matrix_item * 2 >> 7 & 7) << 4) / 2 * 2)));
            float db_df_packed_f32[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&db_df_packed_f32[_pair * 2])[0]), "=f"((&db_df_packed_f32[_pair * 2])[1])
                    : "r"(db_df_packed[_pair]));
            }
            float db_m_packed_f32[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&db_m_packed_f32[_pair * 2])[0]), "=f"((&db_m_packed_f32[_pair * 2])[1])
                    : "r"(db_m_packed[_pair]));
            }
            #pragma unroll
            for (int col_local = 0; col_local < 8; col_local++) {
                float _fma_1 = __fmaf_rn(db_df_packed_f32[col_local], db_m_packed_f32[col_local], db_partial);
                db_partial = _fma_1;
            }
            s_dbeta_partial[warp_id_in_role * 32 + token_col] = db_partial;
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp_id_in_role == 0) {
                float db_value = s_dbeta_partial[token_col] + s_dbeta_partial[32 + token_col] + s_dbeta_partial[64 + token_col] + s_dbeta_partial[96 + token_col];
                long long token = chunk_start + (long long)token_col;
                if (token < eos) {
                    dbeta_active[token * (long long)num_heads + (long long)head] = db_value;
                }
                if (elect_sync()) {
                    mbarrier_arrive(dbeta_done_addr);
                }
            }
            int prep_key_row = prep_tid;
            const int prep_tmem_row_base = warp_id_in_role * 32 << 16;
            float prep_restore = tape_restore_factor[restore_base + (long long)prep_key_row];
            float prep_kr_prefetch = (float)tape_kr[token_key_base + 2048 + (long long)prep_key_row];
            unsigned int _phase_first_outputs_ready_0 = 0;
            mbarrier_wait(first_outputs_ready_addr, _phase_first_outputs_ready_0);
            _phase_first_outputs_ready_0 ^= 1;
            long long prep_out_base = chunk_head * 32 * 128;
            long long prep_chunk_end = chunk_start + 32;
            if (prep_chunk_end > eos) {
                prep_chunk_end = eos;
            }
            int prep_chunk_length = (int)(prep_chunk_end - chunk_start);
            float _exp2_0 = approx_exp2(lower_bound * 1.4426950408889634f * 16.0f);
            float prep_common_factor = _exp2_0;
            const int prep_token_offset = 16;
            float _tmem_load_0[8];
            tmem_ld_x8(&_tmem_load_0[0], taddr + 176 + (unsigned int)prep_tmem_row_base + (unsigned int)prep_token_offset);
            #pragma unroll
            for (int prep_token_local_early = 0; prep_token_local_early < 8; prep_token_local_early++) {
                int prep_token_out_early = prep_token_offset + prep_token_local_early;
                float prep_kr_value_early = prep_kr_prefetch;
                if (prep_token_local_early != 0) {
                    prep_kr_value_early = (float)tape_kr[token_key_base + (long long)(prep_token_out_early * 128) + (long long)prep_key_row];
                }
                int prep_stable_index_early = prep_token_out_early * 128 + prep_key_row;
                s_stable_dkr[prep_stable_index_early] = _tmem_load_0[prep_token_local_early] * prep_kr_value_early;
                _tmem_load_0[prep_token_local_early] = _tmem_load_0[prep_token_local_early] * prep_restore;
            }
            unsigned int _phase_outputs_ready_0 = 0;
            mbarrier_wait(outputs_ready_addr, _phase_outputs_ready_0);
            _phase_outputs_ready_0 ^= 1;
            float _tmem_load_1[8];
            tmem_ld_x8(&_tmem_load_1[0], taddr + 208 + (unsigned int)prep_tmem_row_base + (unsigned int)prep_token_offset);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (int prep_token_local = 0; prep_token_local < 8; prep_token_local++) {
                int prep_token_out = prep_token_offset + prep_token_local;
                long long prep_out_index = prep_out_base + (long long)(prep_token_out * 128) + (long long)prep_key_row;
                int prep_shared_index = prep_token_out * 128 + prep_key_row;
                __nv_bfloat16 prep_i_value_bf16 = s_i[(prep_shared_index * 2 ^ (prep_shared_index * 2 >> 8 & 7) << 4) / 2];
                float _cvt_f32_11 = __bfloat162float(prep_i_value_bf16);
                float prep_i_value = _cvt_f32_11;
                grad_ki[prep_out_index] = _tmem_load_1[prep_token_local] + _tmem_load_0[prep_token_local];
                _tmem_load_1[prep_token_local] = _tmem_load_1[prep_token_local] * prep_i_value;
            }
            float _tmem_load_2[8];
            tmem_ld_x8(&_tmem_load_2[0], taddr + 240 + (unsigned int)prep_tmem_row_base + (unsigned int)prep_token_offset);
            float _tmem_load_3[8];
            tmem_ld_x8(&_tmem_load_3[0], taddr + 272 + (unsigned int)prep_tmem_row_base + (unsigned int)prep_token_offset);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (int prep_token_local2 = 0; prep_token_local2 < 8; prep_token_local2++) {
                int prep_token_out2 = prep_token_offset + prep_token_local2;
                long long prep_out_index2 = prep_out_base + (long long)(prep_token_out2 * 128) + (long long)prep_key_row;
                int prep_shared_index2 = prep_token_out2 * 128 + prep_key_row;
                __nv_bfloat16 prep_q_value_bf16 = s_q[(prep_shared_index2 * 2 ^ (prep_shared_index2 * 2 >> 8 & 7) << 4) / 2];
                float _cvt_f32_12 = __bfloat162float(prep_q_value_bf16);
                float prep_q_value = _cvt_f32_12;
                _tmem_load_3[prep_token_local2] = _tmem_load_3[prep_token_local2] * prep_common_factor;
                float _fma_2 = __fmaf_rn(-_tmem_load_2[prep_token_local2], prep_q_value, _tmem_load_1[prep_token_local2]);
                _tmem_load_1[prep_token_local2] = _fma_2;
                _tmem_load_0[prep_token_local2] = _tmem_load_3[prep_token_local2] * prep_q_value;
                grad_qd[prep_out_index2] = _tmem_load_2[prep_token_local2] + _tmem_load_3[prep_token_local2];
            }
            unsigned int _phase_dbeta_done_0 = 0;
            mbarrier_wait(dbeta_done_addr, _phase_dbeta_done_0);
            _phase_dbeta_done_0 ^= 1;
            float _tmem_load_4[8];
            tmem_ld_x8(&_tmem_load_4[0], taddr + 304 + (unsigned int)prep_tmem_row_base + (unsigned int)prep_token_offset);
            float _tmem_load_5[8];
            tmem_ld_x8(&_tmem_load_5[0], taddr + 336 + (unsigned int)prep_tmem_row_base + (unsigned int)prep_token_offset);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float prep_middle_base_suffix = 0.0f;
            #pragma unroll
            for (int prep_token_local3 = 0; prep_token_local3 < 8; prep_token_local3++) {
                int prep_token_out3 = prep_token_offset + prep_token_local3;
                long long prep_out_index3 = prep_out_base + (long long)(prep_token_out3 * 128) + (long long)prep_key_row;
                int prep_shared_index3 = prep_token_out3 * 128 + prep_key_row;
                __nv_bfloat16 prep_k_value_bf16 = s_k[(prep_shared_index3 * 2 ^ (prep_shared_index3 * 2 >> 8 & 7) << 4) / 2];
                float _cvt_f32_13 = __bfloat162float(prep_k_value_bf16);
                float prep_k_value = _cvt_f32_13;
                _tmem_load_5[prep_token_local3] = _tmem_load_5[prep_token_local3] * (-prep_common_factor);
                float _fma_3 = __fmaf_rn(-_tmem_load_4[prep_token_local3], prep_k_value, _tmem_load_1[prep_token_local3]);
                _tmem_load_1[prep_token_local3] = _fma_3;
                float prep_base_value = _tmem_load_0[prep_token_local3] + _tmem_load_5[prep_token_local3] * prep_k_value;
                int prep_stable_index3 = prep_token_out3 * 128 + prep_key_row;
                s_stable_delta[prep_stable_index3] = _tmem_load_1[prep_token_local3];
                s_stable_base[prep_stable_index3] = prep_base_value;
                if (prep_token_out3 < prep_chunk_length) {
                    prep_middle_base_suffix += prep_base_value;
                }
                grad_kd[prep_out_index3] = _tmem_load_4[prep_token_local3] + _tmem_load_5[prep_token_local3];
            }
            s_middle_base_suffix[prep_key_row] = prep_middle_base_suffix;
            asm volatile("barrier.sync 10, 384;" ::: "memory");
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // epilogue_main
            int ordered_chunk_1 = blockIdx.x / num_heads;
            int chunk_global_1 = consumer_chunk_order[ordered_chunk_1];
            int head_1 = blockIdx.x - ordered_chunk_1 * num_heads;
            int sequence_1 = chunk_sequence[chunk_global_1];
            int local_chunk_1 = chunk_index[chunk_global_1];
            long long bos_1 = cu_seqlens[sequence_1];
            long long eos_1 = cu_seqlens[sequence_1 + 1];
            long long chunk_start_1 = bos_1 + (long long)local_chunk_1 * 32;
            long long chunk_head_1 = (long long)chunk_global_1 * (long long)num_heads + (long long)head_1;
            int warp_id_in_role_1 = (warp - 4);
            int key_row = warp_id_in_role_1 * 32 + lane;
            const int tmem_row_base = warp_id_in_role_1 * 32 << 16;
            long long token_key_base_1 = chunk_head_1 * 32 * 128;
            long long state_base_1 = chunk_head_1 * 128 * 128;
            long long restore_base_1 = chunk_head_1 * 128;
            long long initial_state_base = ((long long)sequence_1 * (long long)num_heads + (long long)head_1) * 128 * 128;
            int checkpoint_needed = 0;
            if (local_chunk_1 != 0) {
                checkpoint_needed = state_checkpoint_needed[chunk_head_1] != 0;
            }
            float restore = tape_restore_factor[restore_base_1 + (long long)key_row];
            float gate_constant = 0.0f;
            unsigned int _phase_recon_output_ready_0 = 0;
            if (local_chunk_1 == 0) {
                #pragma unroll
                for (int value_block = 0; value_block < 128; value_block += 16) {
                    float state_values[16];
                    #pragma unroll
                    for (int value_elem_2 = 0; value_elem_2 < 16; value_elem_2++) {
                        __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(initial_state[initial_state_base + (long long)((value_block + value_elem_2) * 128) + (long long)key_row]);
                        float _cvt_f32_14 = __bfloat162float(_cvt_bf16_8);
                        state_values[value_elem_2] = _cvt_f32_14;
                    }
                    uint32_t state_values_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values[_lp*2 + 0], state_values[_lp*2+1 + 0]));
                        state_values_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(value_block / 2), (const uint32_t*)state_values_bf16);
                }
            } else if (checkpoint_needed != 0) {
                #pragma unroll
                for (int value_block_1 = 0; value_block_1 < 128; value_block_1 += 16) {
                    float state_values_1[16];
                    #pragma unroll
                    for (int value_elem_3 = 0; value_elem_3 < 16; value_elem_3++) {
                        state_values_1[value_elem_3] = (float)chunk_state[state_base_1 + (long long)((value_block_1 + value_elem_3) * 128) + (long long)key_row];
                    }
                    uint32_t state_values_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_values_1[_lp*2 + 0], state_values_1[_lp*2+1 + 0]));
                        state_values_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(value_block_1 / 2), (const uint32_t*)state_values_bf16_1);
                }
            } else {
                long long prev_chunk_head = (long long)(chunk_global_1 - 1) * (long long)num_heads + (long long)head_1;
                long long prev_token_key_base = prev_chunk_head * 32 * 128;
                long long prev_value_token_base = prev_chunk_head * 128 * 32;
                #pragma unroll
                for (int token_half_state = 0; token_half_state < 2; token_half_state++) {
                    int token_offset_state = token_half_state * 16;
                    float kr_values[16];
                    #pragma unroll
                    for (int token_local_state = 0; token_local_state < 16; token_local_state++) {
                        int token_col_state = token_offset_state + token_local_state;
                        kr_values[token_local_state] = (float)tape_kr[prev_token_key_base + (long long)token_col_state * 128 + (long long)key_row];
                    }
                    uint32_t kr_values_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kr_values[_lp*2 + 0], kr_values[_lp*2+1 + 0]));
                        kr_values_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(token_half_state * 8), (const uint32_t*)kr_values_bf16);
                }
                #pragma unroll
                for (int prev_r_segment = 0; prev_r_segment < 4; prev_r_segment++) {
                    float prev_r_values[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(tape_r + prev_value_token_base + (long long)key_row * 32 + (long long)(prev_r_segment * 8));
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&prev_r_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&prev_r_values[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    unsigned int packed_4[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prev_r_values[_lp*2 + 0], prev_r_values[_lp*2+1 + 0]));
                        packed_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((s_prev_r_tc_addr + (unsigned int)(prev_r_segment * 8 / 16 * 4096 + key_row * 32 + prev_r_segment * 8 % 16 * 2 ^ (prev_r_segment * 8 / 16 * 4096 + key_row * 32 + prev_r_segment * 8 % 16 * 2 >> 7 & 1) << 4)) + (unsigned int)(word * 4)), "r"((packed_4[word])));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(recon_inputs_ready_addr);
                }
                mbarrier_wait(recon_output_ready_addr, _phase_recon_output_ready_0);
                _phase_recon_output_ready_0 ^= 1;
                #pragma unroll
                for (int value_block_2 = 0; value_block_2 < 128; value_block_2 += 16) {
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + 368 + (unsigned int)tmem_row_base + (unsigned int)value_block_2);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    uint32_t _tmem_load_6_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                        _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(value_block_2 / 2), (const uint32_t*)_tmem_load_6_bf16);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            if (elect_sync()) {
                mbarrier_arrive(state_tc_ready_addr);
            }
            unsigned int _phase_qki_ready_0 = 0;
            mbarrier_wait(qki_ready_addr, _phase_qki_ready_0);
            _phase_qki_ready_0 ^= 1;
            #pragma unroll
            for (int token_half_ki = 0; token_half_ki < 2; token_half_ki++) {
                int token_offset_ki = token_half_ki * 16;
                float ki_values[16];
                #pragma unroll
                for (int token_local_k = 0; token_local_k < 16; token_local_k++) {
                    int token_col_k = token_offset_ki + token_local_k;
                    int shared_index_k = token_col_k * 128 + key_row;
                    ki_values[token_local_k] = s_k[(shared_index_k * 2 ^ (shared_index_k * 2 >> 8 & 7) << 4) / 2];
                }
                uint32_t ki_values_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_values[_lp*2 + 0], ki_values[_lp*2+1 + 0]));
                    ki_values_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                tmem_st_x8_u32(taddr + 128 + (unsigned int)tmem_row_base + (unsigned int)(token_half_ki * 8), (const uint32_t*)ki_values_bf16);
                #pragma unroll
                for (int token_local_q = 0; token_local_q < 16; token_local_q++) {
                    int token_col_q = token_offset_ki + token_local_q;
                    int shared_index_q = token_col_q * 128 + key_row;
                    ki_values[token_local_q] = s_q[(shared_index_q * 2 ^ (shared_index_q * 2 >> 8 & 7) << 4) / 2];
                }
                uint32_t ki_values_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_values[_lp*2 + 0], ki_values[_lp*2+1 + 0]));
                    ki_values_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                tmem_st_x8_u32(taddr + 144 + (unsigned int)tmem_row_base + (unsigned int)(token_half_ki * 8), (const uint32_t*)ki_values_bf16_0);
                #pragma unroll
                for (int token_local_i = 0; token_local_i < 16; token_local_i++) {
                    int token_col_i = token_offset_ki + token_local_i;
                    int shared_index_i = token_col_i * 128 + key_row;
                    ki_values[token_local_i] = s_i[(shared_index_i * 2 ^ (shared_index_i * 2 >> 8 & 7) << 4) / 2];
                }
                uint32_t ki_values_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_values[_lp*2 + 0], ki_values[_lp*2+1 + 0]));
                    ki_values_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                tmem_st_x8_u32(taddr + 160 + (unsigned int)tmem_row_base + (unsigned int)(token_half_ki * 8), (const uint32_t*)ki_values_bf16_1);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            if (elect_sync()) {
                mbarrier_arrive(qki_tc_ready_addr);
            }
            if (boundary_ready_target != 0) {
                if (warp_id_in_role_1 == 0) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(boundary_ready) + (chunk_head_1);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(boundary_ready_target)) break;
                        }
                    }
                }
                asm volatile("barrier.sync 9, 128;" ::: "memory");
            }
            unsigned int _phase_dh_ready_0 = 0;
            mbarrier_wait(dh_ready_addr, _phase_dh_ready_0);
            _phase_dh_ready_0 ^= 1;
            if (local_chunk_1 == 0) {
                #pragma unroll
                for (int value_block2 = 0; value_block2 < 128; value_block2 += 16) {
                    float dh_values[16];
                    #pragma unroll
                    for (int value_elem2 = 0; value_elem2 < 16; value_elem2++) {
                        int value_row_3 = value_block2 + value_elem2;
                        __nv_bfloat16 _cvt_bf16_9 = __float2bfloat16(initial_state[initial_state_base + (long long)(value_row_3 * 128) + (long long)key_row]);
                        float _cvt_f32_15 = __bfloat162float(_cvt_bf16_9);
                        float state_value = _cvt_f32_15;
                        __nv_bfloat16 dh_value = s_dh_stage[((value_row_3 * 128 + key_row) * 2 ^ ((value_row_3 * 128 + key_row) * 2 >> 7 & 7) << 4) / 2];
                        float _cvt_f32_16 = __bfloat162float(dh_value);
                        dh_values[value_elem2] = _cvt_f32_16;
                        float _fma_4 = __fmaf_rn(dh_values[value_elem2], state_value, gate_constant);
                        gate_constant = _fma_4;
                    }
                    uint32_t dh_values_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh_values[_lp*2 + 0], dh_values[_lp*2+1 + 0]));
                        dh_values_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + (unsigned int)tmem_row_base + (unsigned int)(value_block2 / 2), (const uint32_t*)dh_values_bf16);
                }
            } else if (checkpoint_needed != 0) {
                #pragma unroll
                for (int value_block2_1 = 0; value_block2_1 < 128; value_block2_1 += 16) {
                    float dh_values_1[16];
                    #pragma unroll
                    for (int value_elem2_1 = 0; value_elem2_1 < 16; value_elem2_1++) {
                        int value_row_4 = value_block2_1 + value_elem2_1;
                        float state_value_1 = (float)chunk_state[state_base_1 + (long long)(value_row_4 * 128) + (long long)key_row];
                        __nv_bfloat16 dh_value_1 = s_dh_stage[((value_row_4 * 128 + key_row) * 2 ^ ((value_row_4 * 128 + key_row) * 2 >> 7 & 7) << 4) / 2];
                        float _cvt_f32_17 = __bfloat162float(dh_value_1);
                        dh_values_1[value_elem2_1] = _cvt_f32_17;
                        float _fma_5 = __fmaf_rn(dh_values_1[value_elem2_1], state_value_1, gate_constant);
                        gate_constant = _fma_5;
                    }
                    uint32_t dh_values_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh_values_1[_lp*2 + 0], dh_values_1[_lp*2+1 + 0]));
                        dh_values_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + (unsigned int)tmem_row_base + (unsigned int)(value_block2_1 / 2), (const uint32_t*)dh_values_bf16_1);
                }
            } else {
                #pragma unroll
                for (int value_block2_2 = 0; value_block2_2 < 128; value_block2_2 += 16) {
                    float _tmem_load_7[16];
                    tmem_ld_x16(&_tmem_load_7[0], taddr + 368 + (unsigned int)tmem_row_base + (unsigned int)value_block2_2);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float dh_values_2[16];
                    #pragma unroll
                    for (int value_elem2_2 = 0; value_elem2_2 < 16; value_elem2_2++) {
                        int value_row_5 = value_block2_2 + value_elem2_2;
                        __nv_bfloat16 dh_value_2 = s_dh_stage[((value_row_5 * 128 + key_row) * 2 ^ ((value_row_5 * 128 + key_row) * 2 >> 7 & 7) << 4) / 2];
                        float _cvt_f32_18 = __bfloat162float(dh_value_2);
                        dh_values_2[value_elem2_2] = _cvt_f32_18;
                        float _fma_6 = __fmaf_rn(dh_values_2[value_elem2_2], _tmem_load_7[value_elem2_2], gate_constant);
                        gate_constant = _fma_6;
                    }
                    uint32_t dh_values_bf16_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dh_values_2[_lp*2 + 0], dh_values_2[_lp*2+1 + 0]));
                        dh_values_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    tmem_st_x8_u32(taddr + (unsigned int)tmem_row_base + (unsigned int)(value_block2_2 / 2), (const uint32_t*)dh_values_bf16_2);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float _exp2_1 = approx_exp2(lower_bound * 1.4426950408889634f * 16.0f);
            float common_factor = _exp2_1;
            gate_constant *= common_factor * restore;
            if (elect_sync()) {
                mbarrier_arrive(a_ready_addr);
            }
            float kr_prefetch = (float)tape_kr[token_key_base_1 + (long long)key_row];
            unsigned int _phase_first_outputs_ready_0_1 = 0;
            mbarrier_wait(first_outputs_ready_addr, _phase_first_outputs_ready_0_1);
            _phase_first_outputs_ready_0_1 ^= 1;
            long long out_base = chunk_head_1 * 32 * 128;
            float base_suffix = 0.0f;
            long long chunk_end = chunk_start_1 + 32;
            if (chunk_end > eos_1) {
                chunk_end = eos_1;
            }
            int chunk_length = (int)(chunk_end - chunk_start_1);
            unsigned int _phase_outputs_ready_0_1 = 0;
            unsigned int _phase_dbeta_done_0_1 = 0;
            #pragma unroll
            for (int token_half = 0; token_half < 1; token_half++) {
                int token_offset = token_half * 16;
                float _tmem_load_8[16];
                tmem_ld_x16(&_tmem_load_8[0], taddr + 176 + (unsigned int)tmem_row_base + (unsigned int)token_offset);
                #pragma unroll
                for (int token_local_early = 0; token_local_early < 16; token_local_early++) {
                    int token_out_early = token_offset + token_local_early;
                    float kr_value_early = kr_prefetch;
                    if (token_local_early != 0) {
                        kr_value_early = (float)tape_kr[token_key_base_1 + (long long)(token_out_early * 128) + (long long)key_row];
                    }
                    int stable_index_early = token_out_early * 128 + key_row;
                    s_stable_dkr[stable_index_early] = _tmem_load_8[token_local_early] * kr_value_early;
                    _tmem_load_8[token_local_early] = _tmem_load_8[token_local_early] * restore;
                }
                mbarrier_wait(outputs_ready_addr, _phase_outputs_ready_0_1);
                _phase_outputs_ready_0_1 ^= 1;
                float _tmem_load_9[16];
                tmem_ld_x16(&_tmem_load_9[0], taddr + 208 + (unsigned int)tmem_row_base + (unsigned int)token_offset);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int token_local = 0; token_local < 16; token_local++) {
                    int token_out = token_offset + token_local;
                    long long out_index = out_base + (long long)(token_out * 128) + (long long)key_row;
                    int shared_index = token_out * 128 + key_row;
                    __nv_bfloat16 i_value_bf16 = s_i[(shared_index * 2 ^ (shared_index * 2 >> 8 & 7) << 4) / 2];
                    float _cvt_f32_19 = __bfloat162float(i_value_bf16);
                    float i_value = _cvt_f32_19;
                    grad_ki[out_index] = _tmem_load_9[token_local] + _tmem_load_8[token_local];
                    _tmem_load_9[token_local] = _tmem_load_9[token_local] * i_value;
                }
                float _tmem_load_10[16];
                tmem_ld_x16(&_tmem_load_10[0], taddr + 240 + (unsigned int)tmem_row_base + (unsigned int)token_offset);
                float _tmem_load_11[16];
                tmem_ld_x16(&_tmem_load_11[0], taddr + 272 + (unsigned int)tmem_row_base + (unsigned int)token_offset);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int token_local2 = 0; token_local2 < 16; token_local2++) {
                    int token_out2 = token_offset + token_local2;
                    long long out_index_1 = out_base + (long long)(token_out2 * 128) + (long long)key_row;
                    int shared_index2 = token_out2 * 128 + key_row;
                    __nv_bfloat16 q_value_bf16 = s_q[(shared_index2 * 2 ^ (shared_index2 * 2 >> 8 & 7) << 4) / 2];
                    float _cvt_f32_20 = __bfloat162float(q_value_bf16);
                    float q_value = _cvt_f32_20;
                    _tmem_load_11[token_local2] = _tmem_load_11[token_local2] * common_factor;
                    float _fma_7 = __fmaf_rn(-_tmem_load_10[token_local2], q_value, _tmem_load_9[token_local2]);
                    _tmem_load_9[token_local2] = _fma_7;
                    _tmem_load_8[token_local2] = _tmem_load_11[token_local2] * q_value;
                    grad_qd[out_index_1] = _tmem_load_10[token_local2] + _tmem_load_11[token_local2];
                }
                mbarrier_wait(dbeta_done_addr, _phase_dbeta_done_0_1);
                _phase_dbeta_done_0_1 ^= 1;
                float _tmem_load_12[16];
                tmem_ld_x16(&_tmem_load_12[0], taddr + 304 + (unsigned int)tmem_row_base + (unsigned int)token_offset);
                float _tmem_load_13[16];
                tmem_ld_x16(&_tmem_load_13[0], taddr + 336 + (unsigned int)tmem_row_base + (unsigned int)token_offset);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int token_local3 = 0; token_local3 < 16; token_local3++) {
                    int token_out3 = token_offset + token_local3;
                    long long out_index_2 = out_base + (long long)(token_out3 * 128) + (long long)key_row;
                    int shared_index3 = token_out3 * 128 + key_row;
                    __nv_bfloat16 k_value_bf16 = s_k[(shared_index3 * 2 ^ (shared_index3 * 2 >> 8 & 7) << 4) / 2];
                    float _cvt_f32_21 = __bfloat162float(k_value_bf16);
                    float k_value = _cvt_f32_21;
                    _tmem_load_13[token_local3] = _tmem_load_13[token_local3] * (-common_factor);
                    float _fma_8 = __fmaf_rn(-_tmem_load_12[token_local3], k_value, _tmem_load_9[token_local3]);
                    _tmem_load_9[token_local3] = _fma_8;
                    float base_value = _tmem_load_8[token_local3] + _tmem_load_13[token_local3] * k_value;
                    int stable_index3 = token_out3 * 128 + key_row;
                    s_stable_delta[stable_index3] = _tmem_load_9[token_local3];
                    s_stable_base[stable_index3] = base_value;
                    if (token_out3 < chunk_length) {
                        base_suffix += base_value;
                    }
                    grad_kd[out_index_2] = _tmem_load_12[token_local3] + _tmem_load_13[token_local3];
                }
            }
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            base_suffix += s_middle_base_suffix[key_row] + s_high_base_suffix[key_row];
            float prefix_dkr = 0.0f;
            float crossing = 0.0f;
            #pragma unroll 4
            for (int gate_token = 0; gate_token < chunk_length; gate_token++) {
                int stable_index = gate_token * 128 + key_row;
                long long token_1 = chunk_start_1 + (long long)gate_token;
                long long dlog_index = (token_1 * (long long)num_heads + (long long)head_1) * 128 + (long long)key_row;
                dlog_decay[dlog_index] = base_suffix + gate_constant + (prefix_dkr + crossing);
                crossing += s_stable_delta[stable_index];
                prefix_dkr += s_stable_dkr[stable_index];
                base_suffix -= s_stable_base[stable_index];
            }
            if (elect_sync()) {
                mbarrier_arrive(epilogue_done_addr);
            }
        }
    }
    // ---- Role: mma ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
        { // mma_main
            int ordered_chunk_2 = blockIdx.x / num_heads;
            int chunk_global_2 = consumer_chunk_order[ordered_chunk_2];
            int local_chunk_2 = chunk_index[chunk_global_2];
            int head_2 = blockIdx.x - ordered_chunk_2 * num_heads;
            long long chunk_head_2 = (long long)chunk_global_2 * (long long)num_heads + (long long)head_2;
            int warp_id_in_role_2 = (warp - 8);
            unsigned int _phase_boundary_local_ready_0 = 0;
            unsigned int _phase_dv_ready_0 = 0;
            if (warp_id_in_role_2 != 0) {
                mbarrier_wait(boundary_local_ready_addr, _phase_boundary_local_ready_0);
                _phase_boundary_local_ready_0 ^= 1;
                int sequence_do = chunk_sequence[chunk_global_2];
                long long eos_do = cu_seqlens[sequence_do + 1];
                long long chunk_start_do = cu_seqlens[sequence_do] + (long long)local_chunk_2 * 32;
                int do_tid = (warp_id_in_role_2 - 1) * 32 + lane;
                #pragma unroll 1
                for (int do_group = do_tid; do_group < 512; do_group += 96) {
                    int do_token_col = do_group / 16;
                    int do_segment = do_group % 16;
                    long long token_do = chunk_start_do + (long long)do_token_col;
                    int do_rowmajor_item = do_token_col * 128 + do_segment * 8;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(s_do_stage_addr + (unsigned int)((do_rowmajor_item * 2 ^ (do_rowmajor_item * 2 >> 8 & 7) << 4) / 2 * 2)), "l"(do_ + ((token_do * (long long)num_heads + (long long)head_2) * 128 + (long long)(do_segment * 8))), "r"((token_do < eos_do) ? 16 : 0));
                }
                long long token_value_base_do = chunk_head_2 * 128 * 32;
                #pragma unroll 1
                for (int dr_group = do_tid; dr_group < 512; dr_group += 96) {
                    int dr_copy_item = dr_group * 8;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_drt_addr + (unsigned int)((dr_copy_item * 2 ^ (dr_copy_item * 2 >> 7 & 7) << 4) / 2 * 2)), "l"(chunk_dr + (token_value_base_do + (long long)dr_copy_item)));
                }
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 0;");
                #pragma unroll 1
                for (int do_relay_group = do_tid; do_relay_group < 512; do_relay_group += 96) {
                    int do_relay_token_col = do_relay_group / 16;
                    int do_relay_segment = do_relay_group % 16;
                    int do_relay_rowmajor_item = do_relay_token_col * 128 + do_relay_segment * 8;
                    unsigned int do_packed[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&do_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&do_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&do_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&do_packed[(0) + 3]))
                        : "r"(s_do_stage_addr + (unsigned int)((do_relay_rowmajor_item * 2 ^ (do_relay_rowmajor_item * 2 >> 8 & 7) << 4) / 2 * 2)));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_dot_tc_addr + ((s_dot_tc_addr + (unsigned int)(do_relay_segment * 8 / 64 * 4096 + do_relay_token_col * 128 + do_relay_segment * 8 % 64 * 2 ^ (do_relay_segment * 8 / 64 * 4096 + do_relay_token_col * 128 + do_relay_segment * 8 % 64 * 2 >> 7 & 7) << 4)) - s_dot_tc_addr)), "r"(do_packed[0]), "r"(do_packed[1]), "r"(do_packed[2]), "r"(do_packed[3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(value_tc_ready_addr);
                }
                mbarrier_wait(dv_ready_addr, _phase_dv_ready_0);
                _phase_dv_ready_0 ^= 1;
                int sequence_dv = chunk_sequence[chunk_global_2];
                long long eos_dv = cu_seqlens[sequence_dv + 1];
                long long chunk_start_dv = cu_seqlens[sequence_dv] + (long long)local_chunk_2 * 32;
                int dv_tid = (warp_id_in_role_2 - 1) * 32 + lane;
                #pragma unroll 1
                for (int dv_group = dv_tid; dv_group < 512; dv_group += 96) {
                    int dv_token_col = dv_group / 16;
                    int dv_segment = dv_group % 16;
                    long long token_dv = chunk_start_dv + (long long)dv_token_col;
                    float beta_value_dv = 0.0f;
                    if (token_dv < eos_dv) {
                        beta_value_dv = beta_active[token_dv * (long long)num_heads + (long long)head_2];
                    }
                    float dv_values[8];
                    #pragma unroll
                    for (int value_elem_dv = 0; value_elem_dv < 8; value_elem_dv++) {
                        int value_row_dv = dv_segment * 8 + value_elem_dv;
                        int det_group_dv = dv_segment * 2 + value_elem_dv / 4;
                        float dx_value_dv = s_det[value_row_dv * 32 + (dv_token_col ^ det_group_dv)];
                        dv_values[value_elem_dv] = dx_value_dv * beta_value_dv;
                    }
                    if (token_dv < eos_dv) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(dv_values[0 + 0], dv_values[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(dv_values[0 + 2], dv_values[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(dv_values[0 + 4], dv_values[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(dv_values[0 + 6], dv_values[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dv + ((token_dv * (long long)num_heads + (long long)head_2) * 128 + (long long)(dv_segment * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
                if (elect_sync()) {
                    mbarrier_arrive(dbeta_done_addr);
                }
            }
            unsigned int _phase_recon_inputs_ready_0 = 0;
            unsigned int _phase_state_tc_ready_0 = 0;
            unsigned int _phase_value_tc_ready_0_1 = 0;
            unsigned int _phase_a_ready_0 = 0;
            unsigned int _phase_prep_ready_0 = 0;
            unsigned int _phase_qki_tc_ready_0 = 0;
            if (warp_id_in_role_2 == 0) {
                if (local_chunk_2 != 0) {
                    if (state_checkpoint_needed[chunk_head_2] == 0) {
                        mbarrier_wait(recon_inputs_ready_addr, _phase_recon_inputs_ready_0);
                        _phase_recon_inputs_ready_0 ^= 1;
                        int _mma_b_lo_0 = make_warp_uniform(((s_prev_r_tc_addr) >> 4) & 0x3FFF);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_recon_state), "r"(_mma_b_lo_0), "r"(tmem_tmem_recon_kr), "r"(0));
                        elect_commit(recon_output_ready_addr);
                    }
                }
                mbarrier_wait(dv_ready_addr, _phase_dv_ready_0);
                _phase_dv_ready_0 ^= 1;
                mbarrier_wait(state_tc_ready_addr, _phase_state_tc_ready_0);
                _phase_state_tc_ready_0 ^= 1;
                int _mma_b_lo_1 = make_warp_uniform(((s_det_tc_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dk_boundary), "r"(_mma_b_lo_1), "r"(tmem_tmem_a_state), "r"(0));
                mbarrier_wait(value_tc_ready_addr, _phase_value_tc_ready_0_1);
                _phase_value_tc_ready_0_1 ^= 1;
                int _mma_b_lo_2 = make_warp_uniform(((s_dot_tc_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dq_boundary), "r"(_mma_b_lo_2), "r"(tmem_tmem_a_state), "r"(0));
                mbarrier_wait(a_ready_addr, _phase_a_ready_0);
                _phase_a_ready_0 ^= 1;
                int _mma_b_lo_3 = make_warp_uniform(((s_rt_tc_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dkr), "r"(_mma_b_lo_3), "r"(tmem_tmem_a_dh), "r"(0));
                elect_commit(first_outputs_ready_addr);
                mbarrier_wait(prep_ready_addr, _phase_prep_ready_0);
                _phase_prep_ready_0 ^= 1;
                mbarrier_wait(qki_tc_ready_addr, _phase_qki_tc_ready_0);
                _phase_qki_tc_ready_0 ^= 1;
                int _mma_b_lo_4 = make_warp_uniform(((s_dm_tc_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_di), "r"(_mma_b_lo_4), "r"(tmem_tmem_a_k), "r"(0));
                int _mma_b_lo_5 = make_warp_uniform(((s_dnt_tc_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_di), "r"(_mma_b_lo_5), "r"(tmem_tmem_a_q), "r"(1));
                int _mma_b_lo_6 = make_warp_uniform(((s_dn_tc_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dq_local), "r"(_mma_b_lo_6), "r"(tmem_tmem_a_i), "r"(0));
                int _mma_b_lo_7 = make_warp_uniform(((s_dmt_tc_addr) >> 4) & 0x3FFF);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_dk_local), "r"(_mma_b_lo_7), "r"(tmem_tmem_a_i), "r"(0));
                elect_commit(outputs_ready_addr);
            }
            int key_row_1 = warp_id_in_role_2 * 32 + lane;
            long long restore_base_2 = chunk_head_2 * 128;
            float restore_1 = tape_restore_factor[restore_base_2 + (long long)key_row_1];
            long long token_key_base_2 = chunk_head_2 * 32 * 128;
            float kr_prefetch_1 = (float)tape_kr[token_key_base_2 + 3072 + (long long)key_row_1];
            unsigned int _phase_first_outputs_ready_0_2 = 0;
            mbarrier_wait(first_outputs_ready_addr, _phase_first_outputs_ready_0_2);
            _phase_first_outputs_ready_0_2 ^= 1;
            int sequence_2 = chunk_sequence[chunk_global_2];
            long long bos_2 = cu_seqlens[sequence_2];
            long long eos_2 = cu_seqlens[sequence_2 + 1];
            long long chunk_start_2 = bos_2 + (long long)local_chunk_2 * 32;
            long long chunk_end_1 = chunk_start_2 + 32;
            if (chunk_end_1 > eos_2) {
                chunk_end_1 = eos_2;
            }
            int chunk_length_1 = (int)(chunk_end_1 - chunk_start_2);
            const int tmem_row_base_1 = warp_id_in_role_2 * 32 << 16;
            long long out_base_1 = chunk_head_2 * 32 * 128;
            float _exp2_2 = approx_exp2(lower_bound * 1.4426950408889634f * 16.0f);
            float common_factor_1 = _exp2_2;
            const int token_offset_1 = 24;
            float _tmem_load_14[8];
            tmem_ld_x8(&_tmem_load_14[0], taddr + 176 + (unsigned int)tmem_row_base_1 + (unsigned int)token_offset_1);
            #pragma unroll
            for (int token_local_early_1 = 0; token_local_early_1 < 8; token_local_early_1++) {
                int token_out_early_1 = token_offset_1 + token_local_early_1;
                float kr_value_early_1 = kr_prefetch_1;
                if (token_local_early_1 != 0) {
                    kr_value_early_1 = (float)tape_kr[token_key_base_2 + (long long)(token_out_early_1 * 128) + (long long)key_row_1];
                }
                int stable_index_early_1 = token_out_early_1 * 128 + key_row_1;
                s_stable_dkr[stable_index_early_1] = _tmem_load_14[token_local_early_1] * kr_value_early_1;
                _tmem_load_14[token_local_early_1] = _tmem_load_14[token_local_early_1] * restore_1;
            }
            unsigned int _phase_outputs_ready_0_2 = 0;
            mbarrier_wait(outputs_ready_addr, _phase_outputs_ready_0_2);
            _phase_outputs_ready_0_2 ^= 1;
            float _tmem_load_15[8];
            tmem_ld_x8(&_tmem_load_15[0], taddr + 208 + (unsigned int)tmem_row_base_1 + (unsigned int)token_offset_1);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (int token_local_1 = 0; token_local_1 < 8; token_local_1++) {
                int token_out_1 = token_offset_1 + token_local_1;
                long long out_index_3 = out_base_1 + (long long)(token_out_1 * 128) + (long long)key_row_1;
                int shared_index_1 = token_out_1 * 128 + key_row_1;
                __nv_bfloat16 i_value_bf16_1 = s_i[(shared_index_1 * 2 ^ (shared_index_1 * 2 >> 8 & 7) << 4) / 2];
                float _cvt_f32_22 = __bfloat162float(i_value_bf16_1);
                float i_value_1 = _cvt_f32_22;
                grad_ki[out_index_3] = _tmem_load_15[token_local_1] + _tmem_load_14[token_local_1];
                _tmem_load_15[token_local_1] = _tmem_load_15[token_local_1] * i_value_1;
            }
            float _tmem_load_16[8];
            tmem_ld_x8(&_tmem_load_16[0], taddr + 240 + (unsigned int)tmem_row_base_1 + (unsigned int)token_offset_1);
            float _tmem_load_17[8];
            tmem_ld_x8(&_tmem_load_17[0], taddr + 272 + (unsigned int)tmem_row_base_1 + (unsigned int)token_offset_1);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (int token_local2_1 = 0; token_local2_1 < 8; token_local2_1++) {
                int token_out2_1 = token_offset_1 + token_local2_1;
                long long out_index_4 = out_base_1 + (long long)(token_out2_1 * 128) + (long long)key_row_1;
                int shared_index2_1 = token_out2_1 * 128 + key_row_1;
                __nv_bfloat16 q_value_bf16_1 = s_q[(shared_index2_1 * 2 ^ (shared_index2_1 * 2 >> 8 & 7) << 4) / 2];
                float _cvt_f32_23 = __bfloat162float(q_value_bf16_1);
                float q_value_1 = _cvt_f32_23;
                _tmem_load_17[token_local2_1] = _tmem_load_17[token_local2_1] * common_factor_1;
                float _fma_9 = __fmaf_rn(-_tmem_load_16[token_local2_1], q_value_1, _tmem_load_15[token_local2_1]);
                _tmem_load_15[token_local2_1] = _fma_9;
                _tmem_load_14[token_local2_1] = _tmem_load_17[token_local2_1] * q_value_1;
                grad_qd[out_index_4] = _tmem_load_16[token_local2_1] + _tmem_load_17[token_local2_1];
            }
            unsigned int _phase_dbeta_done_0_2 = 0;
            mbarrier_wait(dbeta_done_addr, _phase_dbeta_done_0_2);
            _phase_dbeta_done_0_2 ^= 1;
            float _tmem_load_18[8];
            tmem_ld_x8(&_tmem_load_18[0], taddr + 304 + (unsigned int)tmem_row_base_1 + (unsigned int)token_offset_1);
            float _tmem_load_19[8];
            tmem_ld_x8(&_tmem_load_19[0], taddr + 336 + (unsigned int)tmem_row_base_1 + (unsigned int)token_offset_1);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float high_base_suffix = 0.0f;
            #pragma unroll
            for (int token_local3_1 = 0; token_local3_1 < 8; token_local3_1++) {
                int token_out3_1 = token_offset_1 + token_local3_1;
                long long out_index_5 = out_base_1 + (long long)(token_out3_1 * 128) + (long long)key_row_1;
                int shared_index3_1 = token_out3_1 * 128 + key_row_1;
                __nv_bfloat16 k_value_bf16_1 = s_k[(shared_index3_1 * 2 ^ (shared_index3_1 * 2 >> 8 & 7) << 4) / 2];
                float _cvt_f32_24 = __bfloat162float(k_value_bf16_1);
                float k_value_1 = _cvt_f32_24;
                _tmem_load_19[token_local3_1] = _tmem_load_19[token_local3_1] * (-common_factor_1);
                float _fma_10 = __fmaf_rn(-_tmem_load_18[token_local3_1], k_value_1, _tmem_load_15[token_local3_1]);
                _tmem_load_15[token_local3_1] = _fma_10;
                float base_value_1 = _tmem_load_14[token_local3_1] + _tmem_load_19[token_local3_1] * k_value_1;
                int stable_index3_1 = token_out3_1 * 128 + key_row_1;
                s_stable_delta[stable_index3_1] = _tmem_load_15[token_local3_1];
                s_stable_base[stable_index3_1] = base_value_1;
                if (token_out3_1 < chunk_length_1) {
                    high_base_suffix += base_value_1;
                }
                grad_kd[out_index_5] = _tmem_load_18[token_local3_1] + _tmem_load_19[token_local3_1];
            }
            s_high_base_suffix[key_row_1] = high_base_suffix;
            asm volatile("barrier.sync 10, 384;" ::: "memory");
            unsigned int _phase_epilogue_done_0 = 0;
            mbarrier_wait(epilogue_done_addr, _phase_epilogue_done_0);
            _phase_epilogue_done_0 ^= 1;
            if (warp_id_in_role_2 == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }

    // Cleanup
}

} // extern "C"

#undef FLASHKDA_INF
#undef NUM_MAIN_STAGES
#undef SMEM_S_DBETA_PARTIAL_OFF
#undef SMEM_S_DBETA_PARTIAL_STAGE_BYTES
#undef SMEM_S_DBETA_PARTIAL_STRIDE
#undef SMEM_S_DET_OFF
#undef SMEM_S_DET_STAGE_BYTES
#undef SMEM_S_DET_STRIDE
#undef SMEM_S_DET_TC_OFF
#undef SMEM_S_DET_TC_STAGE_BYTES
#undef SMEM_S_DET_TC_STRIDE
#undef SMEM_S_DF_OFF
#undef SMEM_S_DF_STAGE_BYTES
#undef SMEM_S_DF_STRIDE
#undef SMEM_S_DH_STAGE_OFF
#undef SMEM_S_DH_STAGE_STAGE_BYTES
#undef SMEM_S_DH_STAGE_STRIDE
#undef SMEM_S_DJ_OFF
#undef SMEM_S_DJ_STAGE_BYTES
#undef SMEM_S_DJ_STRIDE
#undef SMEM_S_DMT_TC_OFF
#undef SMEM_S_DMT_TC_STAGE_BYTES
#undef SMEM_S_DMT_TC_STRIDE
#undef SMEM_S_DM_TC_OFF
#undef SMEM_S_DM_TC_STAGE_BYTES
#undef SMEM_S_DM_TC_STRIDE
#undef SMEM_S_DNT_TC_OFF
#undef SMEM_S_DNT_TC_STAGE_BYTES
#undef SMEM_S_DNT_TC_STRIDE
#undef SMEM_S_DN_TC_OFF
#undef SMEM_S_DN_TC_STAGE_BYTES
#undef SMEM_S_DN_TC_STRIDE
#undef SMEM_S_DOT_OFF
#undef SMEM_S_DOT_STAGE_BYTES
#undef SMEM_S_DOT_STRIDE
#undef SMEM_S_DOT_TC_OFF
#undef SMEM_S_DOT_TC_STAGE_BYTES
#undef SMEM_S_DOT_TC_STRIDE
#undef SMEM_S_DO_STAGE_OFF
#undef SMEM_S_DO_STAGE_STAGE_BYTES
#undef SMEM_S_DO_STAGE_STRIDE
#undef SMEM_S_DRT_OFF
#undef SMEM_S_DRT_STAGE_BYTES
#undef SMEM_S_DRT_STRIDE
#undef SMEM_S_HIGH_BASE_SUFFIX_OFF
#undef SMEM_S_HIGH_BASE_SUFFIX_STAGE_BYTES
#undef SMEM_S_HIGH_BASE_SUFFIX_STRIDE
#undef SMEM_S_I_OFF
#undef SMEM_S_I_STAGE_BYTES
#undef SMEM_S_I_STRIDE
#undef SMEM_S_J_OFF
#undef SMEM_S_J_STAGE_BYTES
#undef SMEM_S_J_STRIDE
#undef SMEM_S_K_OFF
#undef SMEM_S_K_STAGE_BYTES
#undef SMEM_S_K_STRIDE
#undef SMEM_S_MIDDLE_BASE_SUFFIX_OFF
#undef SMEM_S_MIDDLE_BASE_SUFFIX_STAGE_BYTES
#undef SMEM_S_MIDDLE_BASE_SUFFIX_STRIDE
#undef SMEM_S_M_OFF
#undef SMEM_S_M_STAGE_BYTES
#undef SMEM_S_M_STRIDE
#undef SMEM_S_PREV_R_TC_OFF
#undef SMEM_S_PREV_R_TC_STAGE_BYTES
#undef SMEM_S_PREV_R_TC_STRIDE
#undef SMEM_S_Q_OFF
#undef SMEM_S_Q_STAGE_BYTES
#undef SMEM_S_Q_STRIDE
#undef SMEM_S_RT_OFF
#undef SMEM_S_RT_STAGE_BYTES
#undef SMEM_S_RT_STRIDE
#undef SMEM_S_RT_TC_OFF
#undef SMEM_S_RT_TC_STAGE_BYTES
#undef SMEM_S_RT_TC_STRIDE
#undef SMEM_S_STABLE_BASE_OFF
#undef SMEM_S_STABLE_BASE_STAGE_BYTES
#undef SMEM_S_STABLE_BASE_STRIDE
#undef SMEM_S_STABLE_DELTA_OFF
#undef SMEM_S_STABLE_DELTA_STAGE_BYTES
#undef SMEM_S_STABLE_DELTA_STRIDE
#undef SMEM_S_STABLE_DKR_OFF
#undef SMEM_S_STABLE_DKR_STAGE_BYTES
#undef SMEM_S_STABLE_DKR_STRIDE
#undef SMEM_S_TMP_OFF
#undef SMEM_S_TMP_STAGE_BYTES
#undef SMEM_S_TMP_STRIDE
#undef SMEM_S_X_OFF
#undef SMEM_S_X_STAGE_BYTES
#undef SMEM_S_X_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_NCOLS
#undef TMEM_TMEM_A_DH_OFFSET
#undef TMEM_TMEM_A_I_OFFSET
#undef TMEM_TMEM_A_K_OFFSET
#undef TMEM_TMEM_A_Q_OFFSET
#undef TMEM_TMEM_A_STATE_OFFSET
#undef TMEM_TMEM_DI_OFFSET
#undef TMEM_TMEM_DKR_OFFSET
#undef TMEM_TMEM_DK_BOUNDARY_OFFSET
#undef TMEM_TMEM_DK_LOCAL_OFFSET
#undef TMEM_TMEM_DQ_BOUNDARY_OFFSET
#undef TMEM_TMEM_DQ_LOCAL_OFFSET
#undef TMEM_TMEM_RECON_KR_OFFSET
#undef TMEM_TMEM_RECON_STATE_OFFSET
#undef a_ready_addr
#undef boundary_local_ready_addr
#undef dbeta_done_addr
#undef dh_ready_addr
#undef dv_ready_addr
#undef epilogue_done_addr
#undef first_outputs_ready_addr
#undef outputs_ready_addr
#undef prep_ready_addr
#undef qki_ready_addr
#undef qki_tc_ready_addr
#undef recon_inputs_ready_addr
#undef recon_output_ready_addr
#undef s_dbeta_partial_addr
#undef s_det_addr
#undef s_det_tc_addr
#undef s_df_addr
#undef s_dh_stage_addr
#undef s_dj_addr
#undef s_dm_tc_addr
#undef s_dmt_tc_addr
#undef s_dn_tc_addr
#undef s_dnt_tc_addr
#undef s_do_stage_addr
#undef s_dot_addr
#undef s_dot_tc_addr
#undef s_drt_addr
#undef s_high_base_suffix_addr
#undef s_i_addr
#undef s_j_addr
#undef s_k_addr
#undef s_m_addr
#undef s_middle_base_suffix_addr
#undef s_prev_r_tc_addr
#undef s_q_addr
#undef s_rt_addr
#undef s_rt_tc_addr
#undef s_stable_base_addr
#undef s_stable_delta_addr
#undef s_stable_dkr_addr
#undef s_tmp_addr
#undef s_x_addr
#undef state_tc_ready_addr
#undef value_tc_ready_addr

#define FLASHKDA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_flashkda_backward_map_finalize_c32(__nv_bfloat16* __restrict__ decay, __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, float* __restrict__ norm_inv, __nv_bfloat16* __restrict__ g, float* __restrict__ beta_active, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ chunk_sequence, int* __restrict__ chunk_index, int* __restrict__ chunk_pair_start, __nv_bfloat16* __restrict__ grad_qd, __nv_bfloat16* __restrict__ grad_kd, __nv_bfloat16* __restrict__ grad_ki, float* __restrict__ dlog_decay, float* __restrict__ dbeta_active, __nv_bfloat16* __restrict__ dq, __nv_bfloat16* __restrict__ dk, __nv_bfloat16* __restrict__ dg, __nv_bfloat16* __restrict__ dbeta, float* __restrict__ dA_log, float* __restrict__ ddt_bias, int num_pair_heads, int num_heads, float scale, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int block_type = blockIdx.x & 3;
    int task_group = blockIdx.x / 4;
    int warp_id_in_role = (warp - 0);
    int pair_task = task_group * 4 + warp_id_in_role;
    int gate_task = task_group * 8 + (block_type - 2) * 4 + warp_id_in_role;
    int gate_pair_task = gate_task / 2;
    int gate_pair_chunk = gate_task - gate_pair_task * 2;
    if (block_type == 0 && pair_task < num_pair_heads) {
        int pair = pair_task / num_heads;
        int head = pair_task - pair * num_heads;
        int first_chunk = chunk_pair_start[pair];
        int sequence = chunk_sequence[first_chunk];
        int first_local_chunk = chunk_index[first_chunk];
        long long bos = cu_seqlens[sequence];
        long long eos = cu_seqlens[sequence + 1];
        int subwarp = lane / 16;
        int sub_lane = lane - subwarp * 16;
        int elem = sub_lane * 8;
        #pragma unroll
        for (int pair_chunk = 0; pair_chunk < 2; pair_chunk++) {
            int chunk_global = first_chunk + pair_chunk;
            int local_chunk = first_local_chunk + pair_chunk;
            long long chunk_start = bos + (long long)local_chunk * 32;
            if (chunk_start < eos) {
                long long chunk_end = chunk_start + 32;
                if (chunk_end > eos) {
                    chunk_end = eos;
                }
                int chunk_length = (int)(chunk_end - chunk_start);
                long long chunk_head = (long long)chunk_global * (long long)num_heads + (long long)head;
                long long tape_base = chunk_head * 32 * 128;
                #pragma unroll 2
                for (int token_pair = 0; token_pair < chunk_length; token_pair += 2) {
                    int token_col = token_pair + subwarp;
                    long long token = chunk_start + (long long)token_col;
                    long long token_base = (token * (long long)num_heads + (long long)head) * 128;
                    long long index = token_base + (long long)elem;
                    long long tape_index = tape_base + (long long)(token_col * 128) + (long long)elem;
                    float decay_values[8];
                    float grad_values[8];
                    float q_values[8];
                    float q_inv_lane = 0.0f;
                    if (token_col < chunk_length) {
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(decay + index);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&decay_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&decay_values[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(grad_qd + tape_index);
                            uint4 _vld_1[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_1[_blk] = _vptr_1[_blk];
                                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&grad_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&grad_values[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_1[_pair]));
                                }
                            }
                        }
                        {
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(q + index);
                            uint4 _vld_2[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_2[_blk] = _vptr_2[_blk];
                                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&q_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&q_values[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_2[_pair]));
                                }
                            }
                        }
                        if (sub_lane == 0) {
                            long long norm_base = (token * (long long)num_heads + (long long)head) * 2;
                            q_inv_lane = norm_inv[norm_base];
                        }
                    }
                    float _shfl_0 = __shfl_sync(0xFFFFFFFF, q_inv_lane, subwarp * 16);
                    float q_inv = _shfl_0;
                    float q_dot = 0.0f;
                    if (token_col < chunk_length) {
                        #pragma unroll
                        for (int map_i = 0; map_i < 8; map_i++) {
                            grad_values[map_i] = grad_values[map_i] * (decay_values[map_i] * scale);
                            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(q_values[map_i] * q_inv);
                            q_values[map_i] = _cvt_bf16_0;
                            float _fma_0 = __fmaf_rn(grad_values[map_i], q_values[map_i], q_dot);
                            q_dot = _fma_0;
                        }
                    }
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_dot, 8);
                    q_dot += _shfl_xor_0;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, q_dot, 4);
                    q_dot += _shfl_xor_1;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_dot, 2);
                    q_dot += _shfl_xor_2;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, q_dot, 1);
                    q_dot += _shfl_xor_3;
                    if (token_col < chunk_length) {
                        #pragma unroll
                        for (int pullback_i = 0; pullback_i < 8; pullback_i++) {
                            grad_values[pullback_i] = q_inv * (grad_values[pullback_i] - q_values[pullback_i] * q_dot);
                        }
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(grad_values[0 + 0], grad_values[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(grad_values[0 + 2], grad_values[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(grad_values[0 + 4], grad_values[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(grad_values[0 + 6], grad_values[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dq))[index + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
        }
    }
    if (block_type == 1 && pair_task < num_pair_heads) {
        int pair_1 = pair_task / num_heads;
        int head_1 = pair_task - pair_1 * num_heads;
        int first_chunk_1 = chunk_pair_start[pair_1];
        int sequence_1 = chunk_sequence[first_chunk_1];
        int first_local_chunk_1 = chunk_index[first_chunk_1];
        long long bos_1 = cu_seqlens[sequence_1];
        long long eos_1 = cu_seqlens[sequence_1 + 1];
        int subwarp_1 = lane / 16;
        int sub_lane_1 = lane - subwarp_1 * 16;
        int elem_1 = sub_lane_1 * 8;
        #pragma unroll
        for (int pair_chunk_1 = 0; pair_chunk_1 < 2; pair_chunk_1++) {
            int chunk_global_1 = first_chunk_1 + pair_chunk_1;
            int local_chunk_1 = first_local_chunk_1 + pair_chunk_1;
            long long chunk_start_1 = bos_1 + (long long)local_chunk_1 * 32;
            if (chunk_start_1 < eos_1) {
                long long chunk_end_1 = chunk_start_1 + 32;
                if (chunk_end_1 > eos_1) {
                    chunk_end_1 = eos_1;
                }
                int chunk_length_1 = (int)(chunk_end_1 - chunk_start_1);
                long long chunk_head_1 = (long long)chunk_global_1 * (long long)num_heads + (long long)head_1;
                long long tape_base_1 = chunk_head_1 * 32 * 128;
                #pragma unroll 2
                for (int token_pair_1 = 0; token_pair_1 < chunk_length_1; token_pair_1 += 2) {
                    int token_col_1 = token_pair_1 + subwarp_1;
                    long long token_1 = chunk_start_1 + (long long)token_col_1;
                    long long token_base_1 = (token_1 * (long long)num_heads + (long long)head_1) * 128;
                    long long index_1 = token_base_1 + (long long)elem_1;
                    long long tape_index_1 = tape_base_1 + (long long)(token_col_1 * 128) + (long long)elem_1;
                    float decay_values_1[8];
                    float grad_k_values[8];
                    float grad_i_values[8];
                    float k_values[8];
                    float k_inv_lane = 0.0f;
                    if (token_col_1 < chunk_length_1) {
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(decay + index_1);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&decay_values_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&decay_values_1[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_3[_pair]));
                                }
                            }
                        }
                        {
                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(grad_kd + tape_index_1);
                            uint4 _vld_4[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_4[_blk] = _vptr_4[_blk];
                                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&grad_k_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&grad_k_values[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_4[_pair]));
                                }
                            }
                        }
                        {
                            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(grad_ki + tape_index_1);
                            uint4 _vld_5[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_5[_blk] = _vptr_5[_blk];
                                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&grad_i_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&grad_i_values[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_5[_pair]));
                                }
                            }
                        }
                        {
                            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(k + index_1);
                            uint4 _vld_6[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_6[_blk] = _vptr_6[_blk];
                                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&k_values[0 + _blk * 8 + _pair * 2])[0]), "=f"((&k_values[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_6[_pair]));
                                }
                            }
                        }
                        if (sub_lane_1 == 0) {
                            long long norm_base_1 = (token_1 * (long long)num_heads + (long long)head_1) * 2;
                            k_inv_lane = norm_inv[norm_base_1 + 1];
                        }
                    }
                    float _shfl_1 = __shfl_sync(0xFFFFFFFF, k_inv_lane, subwarp_1 * 16);
                    float k_inv = _shfl_1;
                    float k_dot = 0.0f;
                    if (token_col_1 < chunk_length_1) {
                        #pragma unroll
                        for (int map_i_1 = 0; map_i_1 < 8; map_i_1++) {
                            float _rcp_0 = approx_rcp(decay_values_1[map_i_1]);
                            grad_k_values[map_i_1] = grad_i_values[map_i_1] * _rcp_0 + grad_k_values[map_i_1] * decay_values_1[map_i_1];
                            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(k_values[map_i_1] * k_inv);
                            k_values[map_i_1] = _cvt_bf16_1;
                            float _fma_1 = __fmaf_rn(grad_k_values[map_i_1], k_values[map_i_1], k_dot);
                            k_dot = _fma_1;
                        }
                    }
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, k_dot, 8);
                    k_dot += _shfl_xor_4;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_dot, 4);
                    k_dot += _shfl_xor_5;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, k_dot, 2);
                    k_dot += _shfl_xor_6;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, k_dot, 1);
                    k_dot += _shfl_xor_7;
                    if (token_col_1 < chunk_length_1) {
                        #pragma unroll
                        for (int pullback_i_1 = 0; pullback_i_1 < 8; pullback_i_1++) {
                            grad_k_values[pullback_i_1] = k_inv * (grad_k_values[pullback_i_1] - k_values[pullback_i_1] * k_dot);
                        }
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(grad_k_values[0 + 0], grad_k_values[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(grad_k_values[0 + 2], grad_k_values[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(grad_k_values[0 + 4], grad_k_values[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(grad_k_values[0 + 6], grad_k_values[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dk))[index_1 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
        }
    }
    if (block_type >= 2 && gate_pair_task < num_pair_heads) {
        int pair_2 = gate_pair_task / num_heads;
        int head_2 = gate_pair_task - pair_2 * num_heads;
        int first_chunk_2 = chunk_pair_start[pair_2];
        int sequence_2 = chunk_sequence[first_chunk_2];
        int first_local_chunk_2 = chunk_index[first_chunk_2];
        long long bos_2 = cu_seqlens[sequence_2];
        long long eos_2 = cu_seqlens[sequence_2 + 1];
        int elem_2 = lane * 4;
        int pair_chunk_2 = gate_pair_chunk;
        int local_chunk_2 = first_local_chunk_2 + pair_chunk_2;
        long long chunk_start_2 = bos_2 + (long long)local_chunk_2 * 32;
        long long chunk_end_2 = chunk_start_2 + 32;
        if (chunk_end_2 > eos_2) {
            chunk_end_2 = eos_2;
        }
        float gate_a_lane = 0.0f;
        if (lane == 0) {
            float _expf_0 = __expf(A_log[head_2]);
            gate_a_lane = _expf_0;
        }
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, gate_a_lane, 0);
        float gate_a = _shfl_2;
        float gate_scale = lower_bound * gate_a;
        float bias[4];
        float ddt_sum[4];
        float dA_sum[4];
        #pragma unroll
        for (int init_i = 0; init_i < 4; init_i++) {
            bias[init_i] = dt_bias[head_2 * 128 + elem_2 + init_i];
            ddt_sum[init_i] = 0.0f;
            dA_sum[init_i] = 0.0f;
        }
        if (chunk_start_2 < eos_2) {
            int chunk_length_2 = (int)(chunk_end_2 - chunk_start_2);
            #pragma unroll 2
            for (int token_col_2 = 0; token_col_2 < chunk_length_2; token_col_2++) {
                long long token_2 = chunk_start_2 + (long long)token_col_2;
                long long token_base_2 = (token_2 * (long long)num_heads + (long long)head_2) * 128;
                long long index_2 = token_base_2 + (long long)elem_2;
                float g_values[4];
                float dlog_values[4];
                float output_values[4];
                {
                    uint2 _vld_7;
                    _vld_7 = *reinterpret_cast<const uint2*>(g + index_2);
                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&g_values[0 + _pair * 2])[0]), "=f"((&g_values[0 + _pair * 2])[1])
                            : "r"(_vpairs_7[_pair]));
                    }
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dlog_decay + index_2);
                    dlog_values[0 + 0] = _v4.x;
                    dlog_values[0 + 1] = _v4.y;
                    dlog_values[0 + 2] = _v4.z;
                    dlog_values[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int gate_i = 0; gate_i < 4; gate_i++) {
                    float biased = g_values[gate_i] + bias[gate_i];
                    float _expf_1 = __expf((-gate_a) * biased);
                    float sigmoid = 1.0f / (1.0f + _expf_1);
                    float common = dlog_values[gate_i] * gate_scale * sigmoid * (1.0f - sigmoid);
                    output_values[gate_i] = common;
                    ddt_sum[gate_i] = ddt_sum[gate_i] + common;
                    float _fma_2 = __fmaf_rn(common, biased, dA_sum[gate_i]);
                    dA_sum[gate_i] = _fma_2;
                }
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(dg))[index_2]) = _pk2;
                }
                if (lane == 0) {
                    long long beta_index = token_2 * (long long)num_heads + (long long)head_2;
                    float beta_sigmoid = beta_active[beta_index];
                    __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(dbeta_active[beta_index] * beta_sigmoid * (1.0f - beta_sigmoid));
                    dbeta[beta_index] = _cvt_bf16_2;
                }
            }
            #pragma unroll
            for (int publish_i = 0; publish_i < 4; publish_i++) {
                atomicAdd(&ddt_bias[head_2 * 128 + elem_2 + publish_i], ddt_sum[publish_i]);
            }
            float head_dA = 0.0f;
            #pragma unroll
            for (int reduce_i = 0; reduce_i < 4; reduce_i++) {
                head_dA += dA_sum[reduce_i];
            }
            float _warp_reduce_0 = head_dA;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
            head_dA = _warp_reduce_0;
            if (lane == 0) {
                atomicAdd(&dA_log[head_2], head_dA);
            }
        }
    }
}

} // extern "C"

#undef FLASHKDA_INF
#undef NUM_MAIN_STAGES
#undef THREADS

// END FROZEN GENERATED BODY
// clang-format on
