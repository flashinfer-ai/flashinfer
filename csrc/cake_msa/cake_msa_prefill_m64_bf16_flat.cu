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

// clang-format off
// Frozen CAKE-generated CUDA device kernel.
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeMsaTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeMsaTensorMapPack { CakeMsaTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CakeMsaGeneratedTensorMap;

#include <cuda_bf16.h>

#define CAKE_MSA_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_SCORES0_OFFSET 0
#define TMEM_OUTPUT0_OFFSET 128
#define TMEM_SCORES1_OFFSET 256
#define TMEM_OUTPUT1_OFFSET 384
#define NUM_KV_PIPE_STAGES 3
#define SMEM_Q0_SMEM_OFF 1024
#define SMEM_Q0_SMEM_STAGE_BYTES 16384
#define SMEM_Q0_SMEM_STRIDE 16384
#define SMEM_Q1_SMEM_OFF 17408
#define SMEM_Q1_SMEM_STAGE_BYTES 16384
#define SMEM_Q1_SMEM_STRIDE 16384
#define SMEM_KV_SMEM_OFF 33792
#define SMEM_KV_SMEM_STAGE_BYTES 32768
#define SMEM_KV_SMEM_STRIDE 32768
#define SMEM_V_SMEM_OFF 33792
#define SMEM_V_SMEM_STAGE_BYTES 32768
#define SMEM_V_SMEM_STRIDE 32768
#define SMEM_SCALE_SMEM_OFF 132096
#define SMEM_SCALE_SMEM_STAGE_BYTES 1024
#define SMEM_SCALE_SMEM_STRIDE 1024
#define SMEM_MASK_LOW_OFF 134144
#define SMEM_MASK_LOW_STAGE_BYTES 32
#define SMEM_MASK_LOW_STRIDE 32
#define SMEM_MASK_HIGH_OFF 134176
#define SMEM_MASK_HIGH_STAGE_BYTES 32
#define SMEM_MASK_HIGH_STRIDE 32
#define SMEM_UNION_COUNT_SMEM_OFF 134208
#define SMEM_UNION_COUNT_SMEM_STAGE_BYTES 8
#define SMEM_UNION_COUNT_SMEM_STRIDE 8
#define SMEM_UNION_BLOCKS_OFF 134216
#define SMEM_UNION_BLOCKS_STAGE_BYTES 512
#define SMEM_UNION_BLOCKS_STRIDE 512
#define SMEM_TOTAL 134784
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


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
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


__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
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


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_msa_prefill_m64_bf16_flat(const __grid_constant__ CakeMsaTensorMap q_value, const __grid_constant__ CakeMsaTensorMap k_value, const __grid_constant__ CakeMsaTensorMap v_value, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, int* __restrict__ q2k_indices, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ q_offsets, int* __restrict__ kv_lens, int total_q, int num_q_heads, int num_kv_heads, int topk, int batch_size, int causal, int derive_q_offset, float softmax_scale_log2, float lse_temperature_scale, int return_softmax_lse, int return_temperature_lse)
{
    CakeMsaTensorMap const* q = &q_value;
    CakeMsaTensorMap const* k = &k_value;
    CakeMsaTensorMap const* v = &v_value;
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* q0_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q0_smem_addr = smem + 1024;
    __nv_bfloat16* q1_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int q1_smem_addr = smem + 17408;
    __nv_bfloat16* kv_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int kv_smem_addr = smem + 33792;
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int v_smem_addr = smem + 33792;
    float* scale_smem = reinterpret_cast<float*>(smem_raw + 132096);
    const int scale_smem_addr = smem + 132096;
    unsigned int* mask_low = reinterpret_cast<unsigned int*>(smem_raw + 134144);
    const int mask_low_addr = smem + 134144;
    unsigned int* mask_high = reinterpret_cast<unsigned int*>(smem_raw + 134176);
    const int mask_high_addr = smem + 134176;
    int* union_count_smem = reinterpret_cast<int*>(smem_raw + 134208);
    const int union_count_smem_addr = smem + 134208;
    int* union_blocks = reinterpret_cast<int*>(smem_raw + 134216);
    const int union_blocks_addr = smem + 134216;

    // Mbarrier init (10 groups, 19 barriers)
    // Mbarriers at smem_raw[0..152)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // q_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // union_ready: 1 barriers, init_count=32
        mbarrier_init_pred(smem + 8, 32, leader);
        // kv_full: 3 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        mbarrier_init_pred(smem + 24, 1, leader);
        mbarrier_init_pred(smem + 32, 1, leader);
        // --- pipeline 'kv_pipe' ---
        // kv_empty: 3 barriers, init_count=1
        mbarrier_init_pred(smem + 40, 1, leader);
        mbarrier_init_pred(smem + 48, 1, leader);
        mbarrier_init_pred(smem + 56, 1, leader);
        // s_full: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 64, 1, leader);
        mbarrier_init_pred(smem + 72, 1, leader);
        // p_full: 2 barriers, init_count=256
        mbarrier_init_pred(smem + 80, 256, leader);
        mbarrier_init_pred(smem + 88, 256, leader);
        // corr_sig: 2 barriers, init_count=128
        mbarrier_init_pred(smem + 96, 128, leader);
        mbarrier_init_pred(smem + 104, 128, leader);
        // corr_done: 2 barriers, init_count=128
        mbarrier_init_pred(smem + 112, 128, leader);
        mbarrier_init_pred(smem + 120, 128, leader);
        // o_full: 2 barriers, init_count=1
        mbarrier_init_pred(smem + 128, 1, leader);
        mbarrier_init_pred(smem + 136, 1, leader);
        // tmem_dealloc: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 144, 128, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 152);
    if (warp == 0) {
        int _tmem_hold = smem + 152;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define union_ready_addr (mbar_base + 8)
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 40)
    #define s_full_addr (mbar_base + 64)
    #define p_full_addr (mbar_base + 80)
    #define corr_sig_addr (mbar_base + 96)
    #define corr_done_addr (mbar_base + 112)
    #define o_full_addr (mbar_base + 128)
    #define tmem_dealloc_addr (mbar_base + 144)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores0 = taddr;
    const int tmem_output0 = taddr + 128;
    const int tmem_scores1 = taddr + 256;
    const int tmem_output1 = taddr + 384;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            int linear_tile = blockIdx.x;
            int batch = 0;
            int q_tile = 0;
            int tile_prefix = 0;
            int tile_active = 0;
            for (int candidate_batch = 0; candidate_batch < batch_size; candidate_batch++) {
                int candidate_q_begin = cu_seqlens_q[candidate_batch];
                int candidate_q_len = cu_seqlens_q[candidate_batch + 1] - candidate_q_begin;
                int candidate_tiles = (candidate_q_len + 8 - 1) / 8;
                if (linear_tile >= tile_prefix && linear_tile < tile_prefix + candidate_tiles) {
                    batch = candidate_batch;
                    q_tile = linear_tile - tile_prefix;
                    tile_active = 1;
                }
                tile_prefix = tile_prefix + candidate_tiles;
            }
            int kv_head = blockIdx.y;
            int q_head = kv_head * 16;
            int q_begin = cu_seqlens_q[batch];
            int q_len = cu_seqlens_q[batch + 1] - q_begin;
            int q_local_base = q_tile * 8;
            int q_valid = q_len - q_local_base;
            if (q_valid > 8) {
                q_valid = 8;
            }
            if (q_valid < 0) {
                q_valid = 0;
            }
            if (tile_active == 0) {
                q_valid = 0;
            }
            int query_base = q_begin + q_local_base;
            int k_start = cu_seqlens_k[batch];
            int kv_len = cu_seqlens_k[batch + 1] - k_start;
            int query_offset = q_offsets[batch];
            if (derive_q_offset != 0) {
                query_offset = kv_len - q_len;
            }
            int num_n_blocks = (kv_len + 128 - 1) / 128;
            if (causal != 0) {
                int visible_tokens = query_offset + q_local_base + q_valid;
                int visible_blocks = (visible_tokens + 128 - 1) / 128;
                if (num_n_blocks > visible_blocks) {
                    num_n_blocks = visible_blocks;
                }
            }
            if (num_n_blocks < 0) {
                num_n_blocks = 0;
            }
            if (num_n_blocks > 64) {
                num_n_blocks = 64;
            }
            unsigned int _phase_union_ready_0 = 0;
            mbarrier_wait(union_ready_addr, _phase_union_ready_0);
            _phase_union_ready_0 ^= 1;
            int instance = make_warp_uniform(warp / 4);
            int instance_row_offset = make_warp_uniform(instance * 64);
            int instance_token_offset = make_warp_uniform(instance * 4);
            int instance_tmem_offset = make_warp_uniform(instance * 256);
            int union_count = union_count_smem[instance];
            const int warp_in_instance = warp % 4;
            const int tmem_row_origin = warp_in_instance * 32;
            const int logical_row_origin = warp_in_instance * 16;
            int my_row = logical_row_origin + lane % 16;
            int col_half = lane / 16;
            int query_in_instance = my_row / 16;
            int query_in_tile = instance_token_offset + query_in_instance;
            int instance_valid = q_valid - instance_token_offset;
            if (instance_valid > 4) {
                instance_valid = 4;
            }
            if (instance_valid < 0) {
                instance_valid = 0;
            }
            int row_valid = ((query_in_instance < instance_valid) ? 1 : 0);
            unsigned int row_mask_low = mask_low[query_in_tile];
            unsigned int row_mask_high = mask_high[query_in_tile];
            float row_max = -CAKE_MSA_INF;
            float row_sum = 0.0f;
            float temperature_sum = 0.0f;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_s_full_1 = 0;
            unsigned int _phase_corr_done_0 = 0;
            unsigned int _phase_corr_done_1 = 0;
            #pragma unroll 1
            for (int union_index = 0; union_index < union_count; union_index++) {
                int n_block = union_blocks[instance * 64 + union_index];
                if (instance == 0) {
                    mbarrier_wait(s_full_addr, _phase_s_full_0);
                    _phase_s_full_0 ^= 1;
                } else {
                    mbarrier_wait(s_full_addr + 8, _phase_s_full_1);
                    _phase_s_full_1 ^= 1;
                }
                int selected = 0;
                if (row_valid != 0) {
                    if (n_block < 32) {
                        selected = (int)((row_mask_low & (unsigned int)1 << (unsigned int)n_block) != 0);
                    } else {
                        selected = (int)((row_mask_high & (unsigned int)1 << (unsigned int)(n_block - 32)) != 0);
                    }
                }
                int valid_cols = 0;
                if (selected != 0) {
                    valid_cols = kv_len - n_block * 128;
                    if (valid_cols > 128) {
                        valid_cols = 128;
                    }
                    if (causal != 0) {
                        int query_position = query_offset + q_local_base + query_in_tile;
                        int causal_cols = query_position - n_block * 128 + 1;
                        if (valid_cols > causal_cols) {
                            valid_cols = causal_cols;
                        }
                    }
                    if (valid_cols < 0) {
                        valid_cols = 0;
                    }
                }
                int score_addr = taddr + (unsigned int)instance_tmem_offset + (unsigned int)(tmem_row_origin << 16);
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(score_addr)
                    : "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
                    : "r"(score_addr + 32)
                    : "memory");
                int half_valid = valid_cols - col_half * 64;
                if (half_valid < 0) {
                    half_valid = 0;
                }
                if (half_valid > 64) {
                    half_valid = 64;
                }
                if (valid_cols > 0 && half_valid < 64) {
                    uint32_t _slice_lo_mask_0;
                    {
                        int _lim_0 = half_valid;
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
                        if (!(_slice_lo_mask_0 & (1u << _i_1))) _tmem_load_0[0 + _i_1] = -CAKE_MSA_INF;
                    }
                    uint32_t _slice_lo_mask_1;
                    {
                        int _lim_2 = half_valid - 32;
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
                        if (!(_slice_lo_mask_1 & (1u << _i_3))) _tmem_load_0[32 + _i_3] = -CAKE_MSA_INF;
                    }
                }
                float2 _reg_reduce_max2_4 = {-CAKE_MSA_INF, -CAKE_MSA_INF};
                row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_4);
                row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_4);
                float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_4);
                float tile_max = _tmem_load_0_max;
                if (half_valid <= 0) {
                    tile_max = -CAKE_MSA_INF;
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, tile_max, 16);
                float _max_0 = max_noftz(tile_max, _shfl_xor_0);
                tile_max = _max_0;
                float _max_1 = max_noftz(tile_max, row_max);
                float new_max = _max_1;
                float safe_max = ((new_max == -CAKE_MSA_INF) ? 0.0f : new_max);
                float new_max_scaled = safe_max * softmax_scale_log2;
                float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                float acc_scale_log2 = _fma_0;
                float acc_scale;
                float temperature_acc_scale;
                float selected_max;
                if (acc_scale_log2 >= -8.0f) {
                    selected_max = row_max;
                    safe_max = ((row_max == -CAKE_MSA_INF) ? 0.0f : row_max);
                    acc_scale = 1.0f;
                    temperature_acc_scale = 1.0f;
                    new_max_scaled = safe_max * softmax_scale_log2;
                } else {
                    selected_max = new_max;
                    float _exp2_0 = approx_exp2(acc_scale_log2);
                    acc_scale = ((row_max > -CAKE_MSA_INF) ? _exp2_0 : 1.0f);
                    float _exp2_1 = approx_exp2(acc_scale_log2 * lse_temperature_scale);
                    temperature_acc_scale = ((row_max > -CAKE_MSA_INF) ? _exp2_1 : 1.0f);
                }
                row_max = selected_max;
                if (col_half == 0) {
                    scale_smem[instance_row_offset + my_row] = acc_scale;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (instance == 0) {
                    mbarrier_arrive(corr_sig_addr);
                } else {
                    mbarrier_arrive(corr_sig_addr + 8);
                }
                float score_bias = ((valid_cols > 0) ? -new_max_scaled : -CAKE_MSA_INF);
                float block_temperature_sum = 0.0f;
                float block_sum = 0.0f;
                int p_addr = taddr + (unsigned int)instance_tmem_offset + 64 + (unsigned int)(tmem_row_origin << 16);
                if (return_temperature_lse != 0) {
                    const float2 _fma_b2_5 = {softmax_scale_log2 * lse_temperature_scale, softmax_scale_log2 * lse_temperature_scale};
                    const float2 _fma_c2_6 = {score_bias * lse_temperature_scale, score_bias * lse_temperature_scale};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_5, _fma_c2_6);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                    }
                    float2 _reg_reduce_sum2_7 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_7);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_7);
                    float _tmem_load_0_sum = _reg_reduce_sum2_7.x + _reg_reduce_sum2_7.y;
                    float temperature_half = _tmem_load_0_sum;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, temperature_half, 16);
                    block_temperature_sum = temperature_half + _shfl_xor_1;
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                        : "r"(score_addr)
                        : "memory");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                        : "r"(score_addr + 32)
                        : "memory");
                    if (valid_cols > 0 && half_valid < 64) {
                        uint32_t _slice_lo_mask_2;
                        {
                            int _lim_8 = half_valid;
                            if (_lim_8 <= 0) { _slice_lo_mask_2 = 0u; }
                            else if (_lim_8 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_8));
                            }
                        }
                        #pragma unroll
                        for (int _i_9 = 0; _i_9 < 32; _i_9++) {
                            if (!(_slice_lo_mask_2 & (1u << _i_9))) _tmem_load_1[0 + _i_9] = -CAKE_MSA_INF;
                        }
                        uint32_t _slice_lo_mask_3;
                        {
                            int _lim_10 = half_valid - 32;
                            if (_lim_10 <= 0) { _slice_lo_mask_3 = 0u; }
                            else if (_lim_10 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_10));
                            }
                        }
                        #pragma unroll
                        for (int _i_11 = 0; _i_11 < 32; _i_11++) {
                            if (!(_slice_lo_mask_3 & (1u << _i_11))) _tmem_load_1[32 + _i_11] = -CAKE_MSA_INF;
                        }
                    }
                    const float2 _fma_b2_12 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_13 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_lf], _fma_b2_12, _fma_c2_13);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_1[_le] = approx_exp2(_tmem_load_1[_le]);
                    }
                    float2 _reg_reduce_sum2_14 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_1[0], &_reg_reduce_sum2_14);
                    softmax_block_sum(&_tmem_load_1[32], &_reg_reduce_sum2_14);
                    float _tmem_load_1_sum = _reg_reduce_sum2_14.x + _reg_reduce_sum2_14.y;
                    float block_half = _tmem_load_1_sum;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, block_half, 16);
                    block_sum = block_half + _shfl_xor_2;
                    unsigned int packed_p[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        packed_p[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[15]))
                        : "memory");
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 16), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p + 16)[15]))
                        : "memory");
                } else {
                    const float2 _fma_b2_15 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_16 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_15, _fma_c2_16);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                    }
                    float2 _reg_reduce_sum2_17 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_17);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_17);
                    float _tmem_load_0_sum_1 = _reg_reduce_sum2_17.x + _reg_reduce_sum2_17.y;
                    float block_half_1 = _tmem_load_0_sum_1;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, block_half_1, 16);
                    block_sum = block_half_1 + _shfl_xor_3;
                    unsigned int packed_p_1[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        packed_p_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p_1[15]))
                        : "memory");
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                        " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 16), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(packed_p_1 + 16)[15]))
                        : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (instance == 0) {
                    mbarrier_arrive(p_full_addr);
                    mbarrier_wait(corr_done_addr, _phase_corr_done_0);
                    _phase_corr_done_0 ^= 1;
                } else {
                    mbarrier_arrive(p_full_addr + 8);
                    mbarrier_wait(corr_done_addr + 8, _phase_corr_done_1);
                    _phase_corr_done_1 ^= 1;
                }
                row_sum = row_sum * acc_scale + block_sum;
                if (return_temperature_lse != 0) {
                    temperature_sum = temperature_sum * temperature_acc_scale + block_temperature_sum;
                }
            }
            if (col_half == 0) {
                scale_smem[128 + instance_row_offset + my_row] = row_sum;
                scale_smem[256 + instance_row_offset + my_row] = row_max;
                scale_smem[384 + instance_row_offset + my_row] = temperature_sum;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (instance == 0) {
                mbarrier_arrive(corr_sig_addr);
            } else {
                mbarrier_arrive(corr_sig_addr + 8);
            }
        }
    // ---- Role: correction ----
    } else if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // correction_main
            int linear_tile_1 = blockIdx.x;
            int batch_1 = 0;
            int q_tile_1 = 0;
            int tile_prefix_1 = 0;
            int tile_active_1 = 0;
            for (int candidate_batch_1 = 0; candidate_batch_1 < batch_size; candidate_batch_1++) {
                int candidate_q_begin_1 = cu_seqlens_q[candidate_batch_1];
                int candidate_q_len_1 = cu_seqlens_q[candidate_batch_1 + 1] - candidate_q_begin_1;
                int candidate_tiles_1 = (candidate_q_len_1 + 8 - 1) / 8;
                if (linear_tile_1 >= tile_prefix_1 && linear_tile_1 < tile_prefix_1 + candidate_tiles_1) {
                    batch_1 = candidate_batch_1;
                    q_tile_1 = linear_tile_1 - tile_prefix_1;
                    tile_active_1 = 1;
                }
                tile_prefix_1 = tile_prefix_1 + candidate_tiles_1;
            }
            int kv_head_1 = blockIdx.y;
            int q_head_1 = kv_head_1 * 16;
            int q_begin_1 = cu_seqlens_q[batch_1];
            int q_len_1 = cu_seqlens_q[batch_1 + 1] - q_begin_1;
            int q_local_base_1 = q_tile_1 * 8;
            int q_valid_1 = q_len_1 - q_local_base_1;
            if (q_valid_1 > 8) {
                q_valid_1 = 8;
            }
            if (q_valid_1 < 0) {
                q_valid_1 = 0;
            }
            if (tile_active_1 == 0) {
                q_valid_1 = 0;
            }
            int query_base_1 = q_begin_1 + q_local_base_1;
            int k_start_1 = cu_seqlens_k[batch_1];
            int kv_len_1 = cu_seqlens_k[batch_1 + 1] - k_start_1;
            int query_offset_1 = q_offsets[batch_1];
            if (derive_q_offset != 0) {
                query_offset_1 = kv_len_1 - q_len_1;
            }
            int num_n_blocks_1 = (kv_len_1 + 128 - 1) / 128;
            if (causal != 0) {
                int visible_tokens_1 = query_offset_1 + q_local_base_1 + q_valid_1;
                int visible_blocks_1 = (visible_tokens_1 + 128 - 1) / 128;
                if (num_n_blocks_1 > visible_blocks_1) {
                    num_n_blocks_1 = visible_blocks_1;
                }
            }
            if (num_n_blocks_1 < 0) {
                num_n_blocks_1 = 0;
            }
            if (num_n_blocks_1 > 64) {
                num_n_blocks_1 = 64;
            }
            unsigned int _phase_union_ready_0_1 = 0;
            mbarrier_wait(union_ready_addr, _phase_union_ready_0_1);
            _phase_union_ready_0_1 ^= 1;
            int max_union_count = union_count_smem[0];
            int union_count1 = union_count_smem[1];
            if (max_union_count < union_count1) {
                max_union_count = union_count1;
            }
            const int warp_in_role = warp - 8;
            const int tmem_row_origin_1 = warp_in_role * 32;
            const int logical_row_origin_1 = warp_in_role * 16;
            int my_row_1 = logical_row_origin_1 + lane % 16;
            int col_half_1 = lane / 16;
            int row_addr = tmem_row_origin_1 << 16;
            mbarrier_arrive(p_full_addr);
            mbarrier_arrive(p_full_addr + 8);
            unsigned int _phase_corr_sig_0 = 0;
            mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
            _phase_corr_sig_0 ^= 1;
            mbarrier_arrive(corr_done_addr);
            unsigned int _phase_corr_sig_1 = 0;
            mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
            _phase_corr_sig_1 ^= 1;
            mbarrier_arrive(corr_done_addr + 8);
            #pragma unroll 1
            for (int union_index_1 = 1; union_index_1 < max_union_count; union_index_1++) {
                int union_count0 = union_count_smem[0];
                if (union_count0 > union_index_1) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    float acc_scale0 = scale_smem[my_row_1];
                    int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale0 < 1.0f);
                    if (_vote_0 != 0) {
                        float _tmem_load_2[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                            : "r"(taddr + 128 + (unsigned int)row_addr)
                            : "memory");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[63]))
                            : "r"(taddr + 128 + (unsigned int)row_addr + 32)
                            : "memory");
                        const float2 _scale2_0 = {acc_scale0, acc_scale0};
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_0);
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                            " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                            :: "r"(taddr + 128 + (unsigned int)row_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[63]))
                            : "memory");
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr);
                    mbarrier_arrive(corr_done_addr);
                }
                int union_count1_0 = union_count_smem[1];
                if (union_count1_0 > union_index_1) {
                    mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                    _phase_corr_sig_1 ^= 1;
                    float acc_scale1 = scale_smem[64 + my_row_1];
                    int _vote_1 = __any_sync(0xFFFFFFFF, acc_scale1 < 1.0f);
                    if (_vote_1 != 0) {
                        float _tmem_load_3[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                            : "r"(taddr + 256 + 128 + (unsigned int)row_addr)
                            : "memory");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[63]))
                            : "r"(taddr + 256 + 128 + (unsigned int)row_addr + 32)
                            : "memory");
                        const float2 _scale2_1 = {acc_scale1, acc_scale1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_1);
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                            " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                            :: "r"(taddr + 256 + 128 + (unsigned int)row_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_3[63]))
                            : "memory");
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_addr + 8);
                    mbarrier_arrive(corr_done_addr + 8);
                }
            }
            unsigned int _phase_o_full_0 = 0;
            mbarrier_wait(o_full_addr, _phase_o_full_0);
            _phase_o_full_0 ^= 1;
            unsigned int _phase_o_full_1 = 0;
            mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
            _phase_o_full_1 ^= 1;
            mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
            _phase_corr_sig_0 ^= 1;
            mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
            _phase_corr_sig_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            #pragma unroll
            for (int instance_1 = 0; instance_1 < 2; instance_1++) {
                int instance_row_offset_1 = instance_1 * 64;
                int instance_token_offset_1 = instance_1 * 4;
                int output_tmem_offset = instance_1 * 256 + 128;
                float final_sum = scale_smem[128 + instance_row_offset_1 + my_row_1];
                float final_max = scale_smem[256 + instance_row_offset_1 + my_row_1];
                float final_temperature_sum = scale_smem[384 + instance_row_offset_1 + my_row_1];
                float _rcp_0 = approx_rcp(final_sum);
                float inv_sum = ((final_sum > 0.0f && final_sum == final_sum) ? _rcp_0 : 0.0f);
                int query_in_instance_1 = my_row_1 / 16;
                int query_in_tile_1 = instance_token_offset_1 + query_in_instance_1;
                int instance_valid_1 = q_valid_1 - instance_token_offset_1;
                if (instance_valid_1 > 4) {
                    instance_valid_1 = 4;
                }
                if (instance_valid_1 < 0) {
                    instance_valid_1 = 0;
                }
                int output_head = q_head_1 + my_row_1 % 16;
                int query = query_base_1 + query_in_tile_1;
                int output_row = (query * num_q_heads + output_head) * 128;
                float _tmem_load_4[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                    : "r"(taddr + (unsigned int)output_tmem_offset + (unsigned int)row_addr)
                    : "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[63]))
                    : "r"(taddr + (unsigned int)output_tmem_offset + (unsigned int)row_addr + 32)
                    : "memory");
                int col_base = col_half_1 * 64;
                if (query_in_instance_1 < instance_valid_1) {
                    #pragma unroll
                    for (int offset = 0; offset < 64; offset += 8) {
                        {
                            const float2 _prescale2_2 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 4; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[offset])[_ps], _prescale2_2);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                _tmem_load_4[offset + _ps] *= inv_sum;
                            #endif
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_4[offset + 0], _tmem_load_4[offset + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_4[offset + 2], _tmem_load_4[offset + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_4[offset + 4], _tmem_load_4[offset + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_4[offset + 6], _tmem_load_4[offset + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (output_row + col_base + offset)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    if (col_half_1 == 0) {
                        int stat_idx = query * num_q_heads + output_head;
                        if (return_softmax_lse != 0) {
                            float _log2_0;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(final_sum));
                            lse[stat_idx] = ((final_sum > 0.0f) ? final_max * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f : -CAKE_MSA_INF);
                        }
                        if (return_temperature_lse != 0) {
                            float _log2_1;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(final_temperature_sum));
                            temperature_lse[stat_idx] = ((final_temperature_sum > 0.0f) ? final_max * softmax_scale_log2 * 0.6931471805599453f * lse_temperature_scale + _log2_1 * 0.6931471805599453f : -CAKE_MSA_INF);
                        }
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    // ---- Role: mma_warp ----
    } else if (warp == 12) {
        { // mma_warp_main
            unsigned int _phase_union_ready_0_2 = 0;
            mbarrier_wait(union_ready_addr, _phase_union_ready_0_2);
            _phase_union_ready_0_2 ^= 1;
            int max_union_count_1 = union_count_smem[0];
            int union_count1_1 = union_count_smem[1];
            if (max_union_count_1 < union_count1_1) {
                max_union_count_1 = union_count1_1;
            }
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 0;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            int first_pv0 = 1;
            int first_pv1 = 1;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            #pragma unroll 1
            for (int union_index_2 = 0; union_index_2 < max_union_count_1; union_index_2++) {
                int union_count0_1 = union_count_smem[0];
                if (union_count0_1 > union_index_2) {
                    unsigned int k_stage0 = kv_stage;
                    unsigned int k_phase0 = kv_phase;
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (k_stage0) * 8, k_phase0);
                    int _mma_a_lo_0 = make_warp_uniform(((q0_smem_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_0 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage0) * 2048);
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
                    "mov.b32 id, 69207184;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores0), "r"(0));
                    elect_commit(s_full_addr);
                    elect_commit(kv_empty_addr + (k_stage0) * 8);
                }
                int union_count1_0_1 = union_count_smem[1];
                if (union_count1_0_1 > union_index_2) {
                    unsigned int k_stage1 = kv_stage;
                    unsigned int k_phase1 = kv_phase;
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (k_stage1) * 8, k_phase1);
                    int _mma_a_lo_1 = make_warp_uniform(((q1_smem_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_1 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage1) * 2048);
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
                    "mov.b32 id, 69207184;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_scores1), "r"(0));
                    elect_commit(s_full_addr + 8);
                    elect_commit(kv_empty_addr + (k_stage1) * 8);
                }
                if (union_count0_1 > union_index_2) {
                    unsigned int v_stage0 = kv_stage;
                    unsigned int v_phase0 = kv_phase;
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (v_stage0) * 8, v_phase0);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    int _mma_b_lo_2 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage0) * 2048);
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
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output0), "r"(_mma_b_lo_2), "r"(tmem_scores0 + 64), "r"(((first_pv0) ? 0 : 1)));
                    first_pv0 = 0;
                    if (union_index_2 + 1 == union_count0_1) {
                        elect_commit(o_full_addr);
                    }
                    elect_commit(kv_empty_addr + (v_stage0) * 8);
                }
                if (union_count1_0_1 > union_index_2) {
                    unsigned int v_stage1 = kv_stage;
                    unsigned int v_phase1 = kv_phase;
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (v_stage1) * 8, v_phase1);
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    int _mma_b_lo_3 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage1) * 2048);
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
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output1), "r"(_mma_b_lo_3), "r"(tmem_scores1 + 64), "r"(((first_pv1) ? 0 : 1)));
                    first_pv1 = 0;
                    if (union_index_2 + 1 == union_count1_0_1) {
                        elect_commit(o_full_addr + 8);
                    }
                    elect_commit(kv_empty_addr + (v_stage1) * 8);
                }
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        }
    // ---- Role: empty ----
    } else if (warp >= 13 && warp <= 14) {
        // idle — no tasks assigned
    // ---- Role: load_warp ----
    } else if (warp == 15) {
        { // load_warp_main
            int linear_tile_2 = blockIdx.x;
            int batch_2 = 0;
            int q_tile_2 = 0;
            int tile_prefix_2 = 0;
            int tile_active_2 = 0;
            for (int candidate_batch_2 = 0; candidate_batch_2 < batch_size; candidate_batch_2++) {
                int candidate_q_begin_2 = cu_seqlens_q[candidate_batch_2];
                int candidate_q_len_2 = cu_seqlens_q[candidate_batch_2 + 1] - candidate_q_begin_2;
                int candidate_tiles_2 = (candidate_q_len_2 + 8 - 1) / 8;
                if (linear_tile_2 >= tile_prefix_2 && linear_tile_2 < tile_prefix_2 + candidate_tiles_2) {
                    batch_2 = candidate_batch_2;
                    q_tile_2 = linear_tile_2 - tile_prefix_2;
                    tile_active_2 = 1;
                }
                tile_prefix_2 = tile_prefix_2 + candidate_tiles_2;
            }
            int kv_head_2 = blockIdx.y;
            int q_head_2 = kv_head_2 * 16;
            int q_begin_2 = cu_seqlens_q[batch_2];
            int q_len_2 = cu_seqlens_q[batch_2 + 1] - q_begin_2;
            int q_local_base_2 = q_tile_2 * 8;
            int q_valid_2 = q_len_2 - q_local_base_2;
            if (q_valid_2 > 8) {
                q_valid_2 = 8;
            }
            if (q_valid_2 < 0) {
                q_valid_2 = 0;
            }
            if (tile_active_2 == 0) {
                q_valid_2 = 0;
            }
            int query_base_2 = q_begin_2 + q_local_base_2;
            int k_start_2 = cu_seqlens_k[batch_2];
            int kv_len_2 = cu_seqlens_k[batch_2 + 1] - k_start_2;
            int query_offset_2 = q_offsets[batch_2];
            if (derive_q_offset != 0) {
                query_offset_2 = kv_len_2 - q_len_2;
            }
            int num_n_blocks_2 = (kv_len_2 + 128 - 1) / 128;
            if (causal != 0) {
                int visible_tokens_2 = query_offset_2 + q_local_base_2 + q_valid_2;
                int visible_blocks_2 = (visible_tokens_2 + 128 - 1) / 128;
                if (num_n_blocks_2 > visible_blocks_2) {
                    num_n_blocks_2 = visible_blocks_2;
                }
            }
            if (num_n_blocks_2 < 0) {
                num_n_blocks_2 = 0;
            }
            if (num_n_blocks_2 > 64) {
                num_n_blocks_2 = 64;
            }
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 32768);
                tma_4d_gmem2smem(q0_smem_addr, q, 0, q_head_2, query_base_2, 0, q_full_addr);
            }
            int token = lane;
            unsigned int token_mask_low = 0;
            unsigned int token_mask_high = 0;
            if (token < q_valid_2) {
                int selection_base = (kv_head_2 * total_q + query_base_2 + token) * topk;
                #pragma unroll 1
                for (int slot = 0; slot < topk; slot++) {
                    int selected_block = q2k_indices[selection_base + slot];
                    if (selected_block >= 0 && selected_block < num_n_blocks_2) {
                        if (selected_block < 32) {
                            token_mask_low = token_mask_low | (unsigned int)1 << (unsigned int)selected_block;
                        } else {
                            token_mask_high = token_mask_high | (unsigned int)1 << (unsigned int)(selected_block - 32);
                        }
                    }
                }
            }
            if (token < 8) {
                mask_low[token] = token_mask_low;
                mask_high[token] = token_mask_high;
            }
            asm volatile("barrier.sync 8, 32;" ::: "memory");
            if (lane < 2) {
                int instance_2 = lane;
                unsigned int union_low = 0;
                unsigned int union_high = 0;
                #pragma unroll
                for (int query_in_instance_2 = 0; query_in_instance_2 < 4; query_in_instance_2++) {
                    int query_in_tile_2 = instance_2 * 4 + query_in_instance_2;
                    union_low = union_low | mask_low[query_in_tile_2];
                    union_high = union_high | mask_high[query_in_tile_2];
                }
                int union_count_1 = 0;
                int _popc_0 = __popc(union_high);
                int high_count = _popc_0;
                #pragma unroll 1
                for (int _ = 0; _ < high_count; _++) {
                    int _clz_0 = __clz(union_high);
                    int bit = 31 - _clz_0;
                    union_blocks[instance_2 * 64 + union_count_1] = bit + 32;
                    union_high = union_high ^ (unsigned int)1 << (unsigned int)bit;
                    union_count_1 += 1;
                }
                int _popc_1 = __popc(union_low);
                int low_count = _popc_1;
                #pragma unroll 1
                for (int __1 = 0; __1 < low_count; __1++) {
                    int _clz_1 = __clz(union_low);
                    int bit_1 = 31 - _clz_1;
                    union_blocks[instance_2 * 64 + union_count_1] = bit_1;
                    union_low = union_low ^ (unsigned int)1 << (unsigned int)bit_1;
                    union_count_1 += 1;
                }
                if (union_count_1 == 0) {
                    union_blocks[instance_2 * 64] = 0;
                    union_count_1 = 1;
                }
                union_count_smem[instance_2] = union_count_1;
            }
            asm volatile("barrier.sync 8, 32;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(union_ready_addr);
            unsigned int load_stage = 0;
            int max_union_count_2 = union_count_smem[0];
            int union_count1_2 = union_count_smem[1];
            if (max_union_count_2 < union_count1_2) {
                max_union_count_2 = union_count1_2;
            }
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int union_index_3 = 0; union_index_3 < max_union_count_2; union_index_3++) {
                #pragma unroll
                for (int instance_3 = 0; instance_3 < 2; instance_3++) {
                    int instance_union_count = union_count_smem[instance_3];
                    if (instance_union_count > union_index_3) {
                        int n_block_1 = union_blocks[instance_3 * 64 + union_index_3];
                        int token_base = k_start_2 + n_block_1 * 128;
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 32768);
                            int token0 = token_base;
                            int token1 = token_base + 64;
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 32768, k, 0, token0, 0, kv_head_2, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 32768 + 8192, k, 0, token1, 0, kv_head_2, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 32768 + 16384, k, 0, token0, 1, kv_head_2, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 32768 + 24576, k, 0, token1, 1, kv_head_2, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                    }
                }
                #pragma unroll
                for (int instance_4 = 0; instance_4 < 2; instance_4++) {
                    int instance_union_count_1 = union_count_smem[instance_4];
                    if (instance_union_count_1 > union_index_3) {
                        int n_block_2 = union_blocks[instance_4 * 64 + union_index_3];
                        int token_base_1 = k_start_2 + n_block_2 * 128;
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 32768);
                            int token0_1 = token_base_1;
                            int token1_1 = token_base_1 + 64;
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 32768, v, 0, token0_1, 0, kv_head_2, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 32768 + 8192, v, 0, token1_1, 0, kv_head_2, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 32768 + 16384, v, 0, token0_1, 1, kv_head_2, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 32768 + 24576, v, 0, token1_1, 1, kv_head_2, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
// clang-format on
