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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMK_A_OFF 1024
#define SMEM_SMK_A_STAGE_BYTES 8192
#define SMEM_SMK_A_STRIDE 26624
#define SMEM_SMK_B_OFF 9216
#define SMEM_SMK_B_STAGE_BYTES 16384
#define SMEM_SMK_B_STRIDE 26624
#define SMEM_SMK_SF_OFF 25600
#define SMEM_SMK_SF_STAGE_BYTES 2048
#define SMEM_SMK_SF_STRIDE 26624
#define SMEM_SMK_SFX_OFF 54272
#define SMEM_SMK_SFX_STAGE_BYTES 8192
#define SMEM_SMK_SFX_STRIDE 8192
#define SMEM_SMK_PART_OFF 70656
#define SMEM_SMK_PART_STAGE_BYTES 8192
#define SMEM_SMK_PART_STRIDE 8192
#define SMEM_SMK_MAIL_OFF 78848
#define SMEM_SMK_MAIL_STAGE_BYTES 8192
#define SMEM_SMK_MAIL_STRIDE 8192
#define SMEM_TOTAL 87040
#define THREADS 256
#define HAS_ALPHA 1
#define ENABLE_PDL 1
#define LAUNCH_MIN_BLOCKS 2

__device__ __forceinline__ uint32_t cvt_bf16x2_e2m1x2(uint32_t src) {
    uint16_t _fp4_u16 = (uint16_t)(src & 0xFFu);
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
        : "=r"(dst) : "h"(_fp4_u16));
#else
    uint32_t _fp4_f16x2;
    asm("{ .reg .b8 _fp4b, _fp4z0, _fp4z1, _fp4z2;      \n\t"
        "  mov.b32 {_fp4b, _fp4z0, _fp4z1, _fp4z2}, %1;   \n\t"
        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;              }"
        : "=r"(_fp4_f16x2) : "r"((uint32_t)_fp4_u16));
    dst = ((_fp4_f16x2 & 0x7FFF7FFFu) >> 3) | (_fp4_f16x2 & 0x80008000u);
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst) : "r"(dst), "r"(0x77807780u));
#endif
    return dst;
}

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


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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



// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}






extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(1,1,8) void
kernel_cake_blackwell_bf16_fp4_8a6939530b5e50b17f3c(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
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
    #define full_addr (mbar_base + 0)
    #define mail_bar_addr (mbar_base + 16)
    #define sf_full_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = (((blockIdx.z / 8) * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * gridDim.y * (gridDim.z / 8);

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smk_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smk_a_addr = smem + 1024;
    int* smk_b = reinterpret_cast<int*>(smem_raw + 9216);
    const int smk_b_addr = smem + 9216;
    uint8_t* smk_sf = reinterpret_cast<uint8_t*>(smem_raw + 25600);
    const int smk_sf_addr = smem + 25600;
    unsigned int* smk_sfx = reinterpret_cast<unsigned int*>(smem_raw + 54272);
    const int smk_sfx_addr = smem + 54272;
    float* smk_part = reinterpret_cast<float*>(smem_raw + 70656);
    const int smk_part_addr = smem + 70656;
    float* smk_mail = reinterpret_cast<float*>(smem_raw + 78848);
    const int smk_mail_addr = smem + 78848;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // mail_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // sf_full: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 16, 448);
        }
    }

    // === Task calls (declaration order) ===
    int off_n = blockIdx.x * 128;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(sf_full_addr, 4096);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr), "l"((&B_descale)), "r"(off_n), "r"(blockIdx.z * 32),
                   "r"(sf_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr + 1024), "l"((&B_descale)), "r"(off_n + 64), "r"(blockIdx.z * 32),
                   "r"(sf_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr + 26624), "l"((&B_descale)), "r"(off_n), "r"(blockIdx.z * 32 + 16),
                   "r"(sf_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr + 26624 + 1024), "l"((&B_descale)), "r"(off_n + 64), "r"(blockIdx.z * 32 + 16),
                   "r"(sf_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr, 24576);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b_addr), "l"((&B)), "r"(off_n * 2), "r"(blockIdx.z * 32),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b_addr + 8192), "l"((&B)), "r"((off_n + 64) * 2), "r"(blockIdx.z * 32),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr + 8, 24576);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b_addr + 26624), "l"((&B)), "r"(off_n * 2), "r"(blockIdx.z * 32 + 16),
                   "r"(full_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b_addr + 26624 + 8192), "l"((&B)), "r"((off_n + 64) * 2), "r"(blockIdx.z * 32 + 16),
                   "r"(full_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(smk_a_addr), "l"((&A)), "r"(0), "r"(0), "r"(blockIdx.z * 8),
                   "r"(full_addr), "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(smk_a_addr + 26624), "l"((&A)), "r"(0), "r"(0), "r"(blockIdx.z * 8 + 4),
                   "r"(full_addr + 8), "l"(0x14F0000000000000ULL) : "memory");
        }
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
            if (M > 1) {
                mbarrier_expect_tx(mail_bar_addr, 6720);
            }
            mbarrier_arrive(mail_bar_addr);
        }
    }
    unsigned int a_frag[4];
    unsigned int raw[2];
    unsigned int sf_word[4];
    unsigned int carriers[4];
    unsigned int sfx_word[4];
    unsigned int b_frag[2];
    float acc[16];
    acc[0] = 0.0f;
    acc[1] = 0.0f;
    acc[2] = 0.0f;
    acc[3] = 0.0f;
    acc[4] = 0.0f;
    acc[5] = 0.0f;
    acc[6] = 0.0f;
    acc[7] = 0.0f;
    acc[8] = 0.0f;
    acc[9] = 0.0f;
    acc[10] = 0.0f;
    acc[11] = 0.0f;
    acc[12] = 0.0f;
    acc[13] = 0.0f;
    acc[14] = 0.0f;
    acc[15] = 0.0f;
    int warp_0 = warp;
    int nw = warp_0 % 2;
    int pair = warp_0 / 2;
    pair = warp_0 / 2 % 2;
    int npair = warp_0 / 4;
    int pair_off_n = off_n + npair * 64;
    int lane_g = lane / 4;
    int lane_r = lane % 4;
    int a_row = lane % 16;
    int a_col = lane / 16 * 8;
    int b_lane_off = nw * 256 + lane * 8;
    int sf_lane_off = nw * 8 + (lane_g & 4);
    int sf_shift = (lane_g & 3) * 8;
    b_lane_off = b_lane_off + npair * 8192;
    sf_lane_off = sf_lane_off + npair * 1024;
    int sfx_lane_off = (nw * 8 + lane_g) * 16;
    sfx_lane_off = sfx_lane_off + npair * 4096;
    mbarrier_wait(sf_full_addr, 0);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    int sfx_rec = tid;
    int sfx_kb = sfx_rec >> 4;
    int sfx_nlo = sfx_rec & 15;
    int sfx_src = smk_sf_addr + (unsigned int)(sfx_kb * 64) + (unsigned int)(sfx_nlo & 12);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[0])) : "r"(sfx_src));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[1])) : "r"(sfx_src + 16));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[2])) : "r"(sfx_src + 32));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[3])) : "r"(sfx_src + 48));
    int sfx_shift = (sfx_nlo & 3) * 8;
    int sfx_dst = (sfx_kb * 16 + sfx_nlo) * 4;
    uint32_t sfx_carrier;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[0] >> (unsigned int)sfx_shift)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst] = sfx_carrier;
    uint32_t sfx_carrier_1;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[1] >> (unsigned int)sfx_shift)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst + 1] = sfx_carrier_1;
    uint32_t sfx_carrier_2;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[2] >> (unsigned int)sfx_shift)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst + 2] = sfx_carrier_2;
    uint32_t sfx_carrier_3;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[3] >> (unsigned int)sfx_shift)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst + 3] = sfx_carrier_3;
    int sfx_rec_4 = tid + 256;
    int sfx_kb_5 = sfx_rec_4 >> 4;
    int sfx_nlo_6 = sfx_rec_4 & 15;
    int sfx_src_7 = smk_sf_addr + (unsigned int)(sfx_kb_5 * 64) + (unsigned int)(sfx_nlo_6 & 12);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[0])) : "r"(sfx_src_7));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[1])) : "r"(sfx_src_7 + 16));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[2])) : "r"(sfx_src_7 + 32));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[3])) : "r"(sfx_src_7 + 48));
    int sfx_shift_8 = (sfx_nlo_6 & 3) * 8;
    int sfx_dst_9 = (sfx_kb_5 * 16 + sfx_nlo_6) * 4;
    uint32_t sfx_carrier_10;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[0] >> (unsigned int)sfx_shift_8)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_10) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_9] = sfx_carrier_10;
    uint32_t sfx_carrier_11;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[1] >> (unsigned int)sfx_shift_8)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_11) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_9 + 1] = sfx_carrier_11;
    uint32_t sfx_carrier_12;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[2] >> (unsigned int)sfx_shift_8)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_12) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_9 + 2] = sfx_carrier_12;
    uint32_t sfx_carrier_13;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[3] >> (unsigned int)sfx_shift_8)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_13) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_9 + 3] = sfx_carrier_13;
    int sfx_rec_14 = tid;
    int sfx_kb_15 = sfx_rec_14 >> 4;
    int sfx_nlo_16 = sfx_rec_14 & 15;
    int sfx_src_17 = smk_sf_addr + 26624 + (unsigned int)(sfx_kb_15 * 64) + (unsigned int)(sfx_nlo_16 & 12);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[0])) : "r"(sfx_src_17));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[1])) : "r"(sfx_src_17 + 16));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[2])) : "r"(sfx_src_17 + 32));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[3])) : "r"(sfx_src_17 + 48));
    int sfx_shift_18 = (sfx_nlo_16 & 3) * 8;
    int sfx_dst_19 = (sfx_kb_15 * 16 + sfx_nlo_16) * 4 + 2048;
    uint32_t sfx_carrier_20;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[0] >> (unsigned int)sfx_shift_18)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_20) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_19] = sfx_carrier_20;
    uint32_t sfx_carrier_21;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[1] >> (unsigned int)sfx_shift_18)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_21) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_19 + 1] = sfx_carrier_21;
    uint32_t sfx_carrier_22;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[2] >> (unsigned int)sfx_shift_18)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_22) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_19 + 2] = sfx_carrier_22;
    uint32_t sfx_carrier_23;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[3] >> (unsigned int)sfx_shift_18)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_23) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_19 + 3] = sfx_carrier_23;
    int sfx_rec_24 = tid + 256;
    int sfx_kb_25 = sfx_rec_24 >> 4;
    int sfx_nlo_26 = sfx_rec_24 & 15;
    int sfx_src_27 = smk_sf_addr + 26624 + (unsigned int)(sfx_kb_25 * 64) + (unsigned int)(sfx_nlo_26 & 12);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[0])) : "r"(sfx_src_27));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[1])) : "r"(sfx_src_27 + 16));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[2])) : "r"(sfx_src_27 + 32));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[3])) : "r"(sfx_src_27 + 48));
    int sfx_shift_28 = (sfx_nlo_26 & 3) * 8;
    int sfx_dst_29 = (sfx_kb_25 * 16 + sfx_nlo_26) * 4 + 2048;
    uint32_t sfx_carrier_30;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[0] >> (unsigned int)sfx_shift_28)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_30) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_29] = sfx_carrier_30;
    uint32_t sfx_carrier_31;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[1] >> (unsigned int)sfx_shift_28)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_31) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_29 + 1] = sfx_carrier_31;
    uint32_t sfx_carrier_32;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[2] >> (unsigned int)sfx_shift_28)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_32) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_29 + 2] = sfx_carrier_32;
    uint32_t sfx_carrier_33;
    {
        uint32_t _fp4_x16x2 = cvt_bf16x2_e2m1x2(34);
        uint32_t _scale_byte = ((uint32_t)(sfx_word[3] >> (unsigned int)sfx_shift_28)) & 0xFFu;
        uint32_t _scale_f16x2;
        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
        uint32_t _scale_x16x2;
        uint16_t _scale_h0 = (uint16_t)(_scale_f16x2 & 0xFFFFu);
        uint16_t _scale_h1 = (uint16_t)((_scale_f16x2 >> 16) & 0xFFFFu);
        float _scale_f0;
        float _scale_f1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f0) : "h"(_scale_h0));
        asm("cvt.f32.f16 %0, %1;" : "=f"(_scale_f1) : "h"(_scale_h1));
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_scale_x16x2) : "f"(_scale_f1), "f"(_scale_f0));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(sfx_carrier_33) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    smk_sfx[sfx_dst_29 + 3] = sfx_carrier_33;
    __syncthreads();
    mbarrier_wait(full_addr, 0);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)((pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (pair * 16 + a_col) % 64 * 2 ^ ((pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_0[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_0[0]), "=r"(_fp4_bf16x8_0[1]), "=r"(_fp4_bf16x8_0[2]), "=r"(_fp4_bf16x8_0[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_0[0]), "=r"(_fp4_bf16x8_0[1]), "=r"(_fp4_bf16x8_0[2]), "=r"(_fp4_bf16x8_0[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_0[0], hi = (uint16_t)(_fp4_bf16x8_0[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_0[1], hi = (uint16_t)(_fp4_bf16x8_0[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_0[2], hi = (uint16_t)(_fp4_bf16x8_0[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_0[3], hi = (uint16_t)(_fp4_bf16x8_0[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_1[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_1[0]), "=r"(_fp4_bf16x8_1[1]), "=r"(_fp4_bf16x8_1[2]), "=r"(_fp4_bf16x8_1[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_1[0]), "=r"(_fp4_bf16x8_1[1]), "=r"(_fp4_bf16x8_1[2]), "=r"(_fp4_bf16x8_1[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_1[0], hi = (uint16_t)(_fp4_bf16x8_1[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_1[1], hi = (uint16_t)(_fp4_bf16x8_1[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_1[2], hi = (uint16_t)(_fp4_bf16x8_1[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_1[3], hi = (uint16_t)(_fp4_bf16x8_1[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(_fp4_bf16x8_0[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_0;
    uint32_t _bf16x2_mul_1;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(_fp4_bf16x8_0[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_1;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(_fp4_bf16x8_0[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_2;
    uint32_t _bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(_fp4_bf16x8_0[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(_fp4_bf16x8_1[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_4;
    uint32_t _bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(_fp4_bf16x8_1[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_5;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(_fp4_bf16x8_1[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_6;
    uint32_t _bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(_fp4_bf16x8_1[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_7;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)(((2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + ((2 + pair) * 16 + a_col) % 64 * 2 ^ (((2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + ((2 + pair) * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + (unsigned int)((2 + pair) * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + (unsigned int)((2 + pair) * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_2[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_2[0]), "=r"(_fp4_bf16x8_2[1]), "=r"(_fp4_bf16x8_2[2]), "=r"(_fp4_bf16x8_2[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_2[0]), "=r"(_fp4_bf16x8_2[1]), "=r"(_fp4_bf16x8_2[2]), "=r"(_fp4_bf16x8_2[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_2[0], hi = (uint16_t)(_fp4_bf16x8_2[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_2[1], hi = (uint16_t)(_fp4_bf16x8_2[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_2[2], hi = (uint16_t)(_fp4_bf16x8_2[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_2[3], hi = (uint16_t)(_fp4_bf16x8_2[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_3[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_3[0]), "=r"(_fp4_bf16x8_3[1]), "=r"(_fp4_bf16x8_3[2]), "=r"(_fp4_bf16x8_3[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_3[0]), "=r"(_fp4_bf16x8_3[1]), "=r"(_fp4_bf16x8_3[2]), "=r"(_fp4_bf16x8_3[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_3[0], hi = (uint16_t)(_fp4_bf16x8_3[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_3[1], hi = (uint16_t)(_fp4_bf16x8_3[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_3[2], hi = (uint16_t)(_fp4_bf16x8_3[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_3[3], hi = (uint16_t)(_fp4_bf16x8_3[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(_fp4_bf16x8_2[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_8;
    uint32_t _bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(_fp4_bf16x8_2[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_9;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(_fp4_bf16x8_2[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_10;
    uint32_t _bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(_fp4_bf16x8_2[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_11;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(_fp4_bf16x8_3[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_12;
    uint32_t _bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(_fp4_bf16x8_3[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_13;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(_fp4_bf16x8_3[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_14;
    uint32_t _bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(_fp4_bf16x8_3[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_15;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)((64 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (64 + pair * 16 + a_col) % 64 * 2 ^ ((64 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (64 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 2048 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 1024 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_4[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_4[0]), "=r"(_fp4_bf16x8_4[1]), "=r"(_fp4_bf16x8_4[2]), "=r"(_fp4_bf16x8_4[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_4[0]), "=r"(_fp4_bf16x8_4[1]), "=r"(_fp4_bf16x8_4[2]), "=r"(_fp4_bf16x8_4[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_4[0], hi = (uint16_t)(_fp4_bf16x8_4[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_4[1], hi = (uint16_t)(_fp4_bf16x8_4[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_4[2], hi = (uint16_t)(_fp4_bf16x8_4[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_4[3], hi = (uint16_t)(_fp4_bf16x8_4[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_5[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_5[0]), "=r"(_fp4_bf16x8_5[1]), "=r"(_fp4_bf16x8_5[2]), "=r"(_fp4_bf16x8_5[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_5[0]), "=r"(_fp4_bf16x8_5[1]), "=r"(_fp4_bf16x8_5[2]), "=r"(_fp4_bf16x8_5[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_5[0], hi = (uint16_t)(_fp4_bf16x8_5[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_5[1], hi = (uint16_t)(_fp4_bf16x8_5[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_5[2], hi = (uint16_t)(_fp4_bf16x8_5[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_5[3], hi = (uint16_t)(_fp4_bf16x8_5[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(_fp4_bf16x8_4[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_16;
    uint32_t _bf16x2_mul_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(_fp4_bf16x8_4[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_17;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(_fp4_bf16x8_4[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_18;
    uint32_t _bf16x2_mul_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(_fp4_bf16x8_4[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_19;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(_fp4_bf16x8_5[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_20;
    uint32_t _bf16x2_mul_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(_fp4_bf16x8_5[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_21;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(_fp4_bf16x8_5[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_22;
    uint32_t _bf16x2_mul_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(_fp4_bf16x8_5[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_23;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)((64 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (64 + (2 + pair) * 16 + a_col) % 64 * 2 ^ ((64 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (64 + (2 + pair) * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 2048 + (unsigned int)((2 + pair) * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 1024 + (unsigned int)((2 + pair) * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_6[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_6[0]), "=r"(_fp4_bf16x8_6[1]), "=r"(_fp4_bf16x8_6[2]), "=r"(_fp4_bf16x8_6[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_6[0]), "=r"(_fp4_bf16x8_6[1]), "=r"(_fp4_bf16x8_6[2]), "=r"(_fp4_bf16x8_6[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_6[0], hi = (uint16_t)(_fp4_bf16x8_6[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_6[1], hi = (uint16_t)(_fp4_bf16x8_6[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_6[2], hi = (uint16_t)(_fp4_bf16x8_6[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_6[3], hi = (uint16_t)(_fp4_bf16x8_6[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_7[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_7[0]), "=r"(_fp4_bf16x8_7[1]), "=r"(_fp4_bf16x8_7[2]), "=r"(_fp4_bf16x8_7[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_7[0]), "=r"(_fp4_bf16x8_7[1]), "=r"(_fp4_bf16x8_7[2]), "=r"(_fp4_bf16x8_7[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_7[0], hi = (uint16_t)(_fp4_bf16x8_7[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_7[1], hi = (uint16_t)(_fp4_bf16x8_7[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_7[2], hi = (uint16_t)(_fp4_bf16x8_7[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_7[3], hi = (uint16_t)(_fp4_bf16x8_7[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(_fp4_bf16x8_6[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_24;
    uint32_t _bf16x2_mul_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(_fp4_bf16x8_6[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_25;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(_fp4_bf16x8_6[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_26;
    uint32_t _bf16x2_mul_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(_fp4_bf16x8_6[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_27;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(_fp4_bf16x8_7[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_28;
    uint32_t _bf16x2_mul_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(_fp4_bf16x8_7[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_29;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(_fp4_bf16x8_7[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_30;
    uint32_t _bf16x2_mul_31;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(_fp4_bf16x8_7[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_31;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)((128 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (128 + pair * 16 + a_col) % 64 * 2 ^ ((128 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (128 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 4096 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 2048 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_8[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_8[0]), "=r"(_fp4_bf16x8_8[1]), "=r"(_fp4_bf16x8_8[2]), "=r"(_fp4_bf16x8_8[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_8[0]), "=r"(_fp4_bf16x8_8[1]), "=r"(_fp4_bf16x8_8[2]), "=r"(_fp4_bf16x8_8[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_8[0], hi = (uint16_t)(_fp4_bf16x8_8[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_8[1], hi = (uint16_t)(_fp4_bf16x8_8[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_8[2], hi = (uint16_t)(_fp4_bf16x8_8[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_8[3], hi = (uint16_t)(_fp4_bf16x8_8[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_9[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_9[0]), "=r"(_fp4_bf16x8_9[1]), "=r"(_fp4_bf16x8_9[2]), "=r"(_fp4_bf16x8_9[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_9[0]), "=r"(_fp4_bf16x8_9[1]), "=r"(_fp4_bf16x8_9[2]), "=r"(_fp4_bf16x8_9[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_9[0], hi = (uint16_t)(_fp4_bf16x8_9[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_9[1], hi = (uint16_t)(_fp4_bf16x8_9[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_9[2], hi = (uint16_t)(_fp4_bf16x8_9[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_9[3], hi = (uint16_t)(_fp4_bf16x8_9[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_32;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(_fp4_bf16x8_8[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_32;
    uint32_t _bf16x2_mul_33;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(_fp4_bf16x8_8[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_33;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_34;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(_fp4_bf16x8_8[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_34;
    uint32_t _bf16x2_mul_35;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(_fp4_bf16x8_8[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_35;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_36;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(_fp4_bf16x8_9[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_36;
    uint32_t _bf16x2_mul_37;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(_fp4_bf16x8_9[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_37;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_38;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(_fp4_bf16x8_9[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_38;
    uint32_t _bf16x2_mul_39;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(_fp4_bf16x8_9[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_39;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)((128 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (128 + (2 + pair) * 16 + a_col) % 64 * 2 ^ ((128 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (128 + (2 + pair) * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 4096 + (unsigned int)((2 + pair) * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 2048 + (unsigned int)((2 + pair) * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_10[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_10[0]), "=r"(_fp4_bf16x8_10[1]), "=r"(_fp4_bf16x8_10[2]), "=r"(_fp4_bf16x8_10[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_10[0]), "=r"(_fp4_bf16x8_10[1]), "=r"(_fp4_bf16x8_10[2]), "=r"(_fp4_bf16x8_10[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_10[0], hi = (uint16_t)(_fp4_bf16x8_10[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_10[1], hi = (uint16_t)(_fp4_bf16x8_10[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_10[2], hi = (uint16_t)(_fp4_bf16x8_10[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_10[3], hi = (uint16_t)(_fp4_bf16x8_10[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_11[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_11[0]), "=r"(_fp4_bf16x8_11[1]), "=r"(_fp4_bf16x8_11[2]), "=r"(_fp4_bf16x8_11[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_11[0]), "=r"(_fp4_bf16x8_11[1]), "=r"(_fp4_bf16x8_11[2]), "=r"(_fp4_bf16x8_11[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_11[0], hi = (uint16_t)(_fp4_bf16x8_11[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_11[1], hi = (uint16_t)(_fp4_bf16x8_11[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_11[2], hi = (uint16_t)(_fp4_bf16x8_11[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_11[3], hi = (uint16_t)(_fp4_bf16x8_11[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_40;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(_fp4_bf16x8_10[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_40;
    uint32_t _bf16x2_mul_41;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(_fp4_bf16x8_10[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_41;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_42;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(_fp4_bf16x8_10[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_42;
    uint32_t _bf16x2_mul_43;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(_fp4_bf16x8_10[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_43;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_44;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(_fp4_bf16x8_11[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_44;
    uint32_t _bf16x2_mul_45;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(_fp4_bf16x8_11[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_45;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_46;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(_fp4_bf16x8_11[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_46;
    uint32_t _bf16x2_mul_47;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(_fp4_bf16x8_11[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_47;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)((192 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (192 + pair * 16 + a_col) % 64 * 2 ^ ((192 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (192 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 6144 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 3072 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_12[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_12[0]), "=r"(_fp4_bf16x8_12[1]), "=r"(_fp4_bf16x8_12[2]), "=r"(_fp4_bf16x8_12[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_12[0]), "=r"(_fp4_bf16x8_12[1]), "=r"(_fp4_bf16x8_12[2]), "=r"(_fp4_bf16x8_12[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_12[0], hi = (uint16_t)(_fp4_bf16x8_12[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_12[1], hi = (uint16_t)(_fp4_bf16x8_12[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_12[2], hi = (uint16_t)(_fp4_bf16x8_12[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_12[3], hi = (uint16_t)(_fp4_bf16x8_12[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_13[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_13[0]), "=r"(_fp4_bf16x8_13[1]), "=r"(_fp4_bf16x8_13[2]), "=r"(_fp4_bf16x8_13[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_13[0]), "=r"(_fp4_bf16x8_13[1]), "=r"(_fp4_bf16x8_13[2]), "=r"(_fp4_bf16x8_13[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_13[0], hi = (uint16_t)(_fp4_bf16x8_13[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_13[1], hi = (uint16_t)(_fp4_bf16x8_13[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_13[2], hi = (uint16_t)(_fp4_bf16x8_13[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_13[3], hi = (uint16_t)(_fp4_bf16x8_13[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_48;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(_fp4_bf16x8_12[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_48;
    uint32_t _bf16x2_mul_49;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(_fp4_bf16x8_12[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_49;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_50;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(_fp4_bf16x8_12[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_50;
    uint32_t _bf16x2_mul_51;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(_fp4_bf16x8_12[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_51;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_52;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(_fp4_bf16x8_13[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_52;
    uint32_t _bf16x2_mul_53;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(_fp4_bf16x8_13[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_53;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_54;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(_fp4_bf16x8_13[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_54;
    uint32_t _bf16x2_mul_55;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(_fp4_bf16x8_13[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_55;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)((192 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (192 + (2 + pair) * 16 + a_col) % 64 * 2 ^ ((192 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (192 + (2 + pair) * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 6144 + (unsigned int)((2 + pair) * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 3072 + (unsigned int)((2 + pair) * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_14[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_14[0]), "=r"(_fp4_bf16x8_14[1]), "=r"(_fp4_bf16x8_14[2]), "=r"(_fp4_bf16x8_14[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_14[0]), "=r"(_fp4_bf16x8_14[1]), "=r"(_fp4_bf16x8_14[2]), "=r"(_fp4_bf16x8_14[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_14[0], hi = (uint16_t)(_fp4_bf16x8_14[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_14[1], hi = (uint16_t)(_fp4_bf16x8_14[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_14[2], hi = (uint16_t)(_fp4_bf16x8_14[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_14[3], hi = (uint16_t)(_fp4_bf16x8_14[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_15[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_15[0]), "=r"(_fp4_bf16x8_15[1]), "=r"(_fp4_bf16x8_15[2]), "=r"(_fp4_bf16x8_15[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_15[0]), "=r"(_fp4_bf16x8_15[1]), "=r"(_fp4_bf16x8_15[2]), "=r"(_fp4_bf16x8_15[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_15[0], hi = (uint16_t)(_fp4_bf16x8_15[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_15[1], hi = (uint16_t)(_fp4_bf16x8_15[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_15[2], hi = (uint16_t)(_fp4_bf16x8_15[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_15[3], hi = (uint16_t)(_fp4_bf16x8_15[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_56;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(_fp4_bf16x8_14[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_56;
    uint32_t _bf16x2_mul_57;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(_fp4_bf16x8_14[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_57;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_58;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(_fp4_bf16x8_14[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_58;
    uint32_t _bf16x2_mul_59;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(_fp4_bf16x8_14[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_59;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_60;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(_fp4_bf16x8_15[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_60;
    uint32_t _bf16x2_mul_61;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(_fp4_bf16x8_15[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_61;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_62;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(_fp4_bf16x8_15[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_62;
    uint32_t _bf16x2_mul_63;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(_fp4_bf16x8_15[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_63;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    mbarrier_wait(full_addr + 8, 0);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 26624 + (unsigned int)((pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (pair * 16 + a_col) % 64 * 2 ^ ((pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 26624 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 8192 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_16[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_16[0]), "=r"(_fp4_bf16x8_16[1]), "=r"(_fp4_bf16x8_16[2]), "=r"(_fp4_bf16x8_16[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_16[0]), "=r"(_fp4_bf16x8_16[1]), "=r"(_fp4_bf16x8_16[2]), "=r"(_fp4_bf16x8_16[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_16[0], hi = (uint16_t)(_fp4_bf16x8_16[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_16[1], hi = (uint16_t)(_fp4_bf16x8_16[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_16[2], hi = (uint16_t)(_fp4_bf16x8_16[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_16[3], hi = (uint16_t)(_fp4_bf16x8_16[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_17[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_17[0]), "=r"(_fp4_bf16x8_17[1]), "=r"(_fp4_bf16x8_17[2]), "=r"(_fp4_bf16x8_17[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_17[0]), "=r"(_fp4_bf16x8_17[1]), "=r"(_fp4_bf16x8_17[2]), "=r"(_fp4_bf16x8_17[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_17[0], hi = (uint16_t)(_fp4_bf16x8_17[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_17[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_17[1], hi = (uint16_t)(_fp4_bf16x8_17[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_17[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_17[2], hi = (uint16_t)(_fp4_bf16x8_17[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_17[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_17[3], hi = (uint16_t)(_fp4_bf16x8_17[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_17[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_64;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_64) : "r"(_fp4_bf16x8_16[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_64;
    uint32_t _bf16x2_mul_65;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_65) : "r"(_fp4_bf16x8_16[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_65;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_66;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_66) : "r"(_fp4_bf16x8_16[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_66;
    uint32_t _bf16x2_mul_67;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_67) : "r"(_fp4_bf16x8_16[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_67;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_68;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_68) : "r"(_fp4_bf16x8_17[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_68;
    uint32_t _bf16x2_mul_69;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_69) : "r"(_fp4_bf16x8_17[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_69;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_70;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_70) : "r"(_fp4_bf16x8_17[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_70;
    uint32_t _bf16x2_mul_71;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_71) : "r"(_fp4_bf16x8_17[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_71;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 26624 + (unsigned int)(((2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + ((2 + pair) * 16 + a_col) % 64 * 2 ^ (((2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + ((2 + pair) * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 26624 + (unsigned int)((2 + pair) * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 8192 + (unsigned int)((2 + pair) * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_18[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_18[0]), "=r"(_fp4_bf16x8_18[1]), "=r"(_fp4_bf16x8_18[2]), "=r"(_fp4_bf16x8_18[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_18[0]), "=r"(_fp4_bf16x8_18[1]), "=r"(_fp4_bf16x8_18[2]), "=r"(_fp4_bf16x8_18[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_18[0], hi = (uint16_t)(_fp4_bf16x8_18[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_18[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_18[1], hi = (uint16_t)(_fp4_bf16x8_18[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_18[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_18[2], hi = (uint16_t)(_fp4_bf16x8_18[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_18[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_18[3], hi = (uint16_t)(_fp4_bf16x8_18[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_18[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_19[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_19[0]), "=r"(_fp4_bf16x8_19[1]), "=r"(_fp4_bf16x8_19[2]), "=r"(_fp4_bf16x8_19[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_19[0]), "=r"(_fp4_bf16x8_19[1]), "=r"(_fp4_bf16x8_19[2]), "=r"(_fp4_bf16x8_19[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_19[0], hi = (uint16_t)(_fp4_bf16x8_19[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_19[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_19[1], hi = (uint16_t)(_fp4_bf16x8_19[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_19[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_19[2], hi = (uint16_t)(_fp4_bf16x8_19[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_19[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_19[3], hi = (uint16_t)(_fp4_bf16x8_19[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_19[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_72;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_72) : "r"(_fp4_bf16x8_18[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_72;
    uint32_t _bf16x2_mul_73;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_73) : "r"(_fp4_bf16x8_18[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_73;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_74;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_74) : "r"(_fp4_bf16x8_18[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_74;
    uint32_t _bf16x2_mul_75;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_75) : "r"(_fp4_bf16x8_18[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_75;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_76;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_76) : "r"(_fp4_bf16x8_19[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_76;
    uint32_t _bf16x2_mul_77;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_77) : "r"(_fp4_bf16x8_19[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_77;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_78;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_78) : "r"(_fp4_bf16x8_19[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_78;
    uint32_t _bf16x2_mul_79;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_79) : "r"(_fp4_bf16x8_19[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_79;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 26624 + (unsigned int)((64 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (64 + pair * 16 + a_col) % 64 * 2 ^ ((64 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (64 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 26624 + 2048 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 8192 + 1024 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_20[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_20[0]), "=r"(_fp4_bf16x8_20[1]), "=r"(_fp4_bf16x8_20[2]), "=r"(_fp4_bf16x8_20[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_20[0]), "=r"(_fp4_bf16x8_20[1]), "=r"(_fp4_bf16x8_20[2]), "=r"(_fp4_bf16x8_20[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_20[0], hi = (uint16_t)(_fp4_bf16x8_20[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_20[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_20[1], hi = (uint16_t)(_fp4_bf16x8_20[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_20[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_20[2], hi = (uint16_t)(_fp4_bf16x8_20[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_20[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_20[3], hi = (uint16_t)(_fp4_bf16x8_20[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_20[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_21[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_21[0]), "=r"(_fp4_bf16x8_21[1]), "=r"(_fp4_bf16x8_21[2]), "=r"(_fp4_bf16x8_21[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_21[0]), "=r"(_fp4_bf16x8_21[1]), "=r"(_fp4_bf16x8_21[2]), "=r"(_fp4_bf16x8_21[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_21[0], hi = (uint16_t)(_fp4_bf16x8_21[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_21[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_21[1], hi = (uint16_t)(_fp4_bf16x8_21[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_21[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_21[2], hi = (uint16_t)(_fp4_bf16x8_21[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_21[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_21[3], hi = (uint16_t)(_fp4_bf16x8_21[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_21[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_80;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_80) : "r"(_fp4_bf16x8_20[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_80;
    uint32_t _bf16x2_mul_81;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_81) : "r"(_fp4_bf16x8_20[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_81;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_82;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_82) : "r"(_fp4_bf16x8_20[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_82;
    uint32_t _bf16x2_mul_83;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_83) : "r"(_fp4_bf16x8_20[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_83;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_84;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_84) : "r"(_fp4_bf16x8_21[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_84;
    uint32_t _bf16x2_mul_85;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_85) : "r"(_fp4_bf16x8_21[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_85;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_86;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_86) : "r"(_fp4_bf16x8_21[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_86;
    uint32_t _bf16x2_mul_87;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_87) : "r"(_fp4_bf16x8_21[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_87;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 26624 + (unsigned int)((64 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (64 + (2 + pair) * 16 + a_col) % 64 * 2 ^ ((64 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (64 + (2 + pair) * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 26624 + 2048 + (unsigned int)((2 + pair) * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 8192 + 1024 + (unsigned int)((2 + pair) * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_22[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_22[0]), "=r"(_fp4_bf16x8_22[1]), "=r"(_fp4_bf16x8_22[2]), "=r"(_fp4_bf16x8_22[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_22[0]), "=r"(_fp4_bf16x8_22[1]), "=r"(_fp4_bf16x8_22[2]), "=r"(_fp4_bf16x8_22[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_22[0], hi = (uint16_t)(_fp4_bf16x8_22[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_22[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_22[1], hi = (uint16_t)(_fp4_bf16x8_22[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_22[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_22[2], hi = (uint16_t)(_fp4_bf16x8_22[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_22[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_22[3], hi = (uint16_t)(_fp4_bf16x8_22[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_22[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_23[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_23[0]), "=r"(_fp4_bf16x8_23[1]), "=r"(_fp4_bf16x8_23[2]), "=r"(_fp4_bf16x8_23[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_23[0]), "=r"(_fp4_bf16x8_23[1]), "=r"(_fp4_bf16x8_23[2]), "=r"(_fp4_bf16x8_23[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_23[0], hi = (uint16_t)(_fp4_bf16x8_23[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_23[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_23[1], hi = (uint16_t)(_fp4_bf16x8_23[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_23[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_23[2], hi = (uint16_t)(_fp4_bf16x8_23[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_23[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_23[3], hi = (uint16_t)(_fp4_bf16x8_23[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_23[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_88;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_88) : "r"(_fp4_bf16x8_22[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_88;
    uint32_t _bf16x2_mul_89;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_89) : "r"(_fp4_bf16x8_22[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_89;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_90;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_90) : "r"(_fp4_bf16x8_22[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_90;
    uint32_t _bf16x2_mul_91;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_91) : "r"(_fp4_bf16x8_22[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_91;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_92;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_92) : "r"(_fp4_bf16x8_23[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_92;
    uint32_t _bf16x2_mul_93;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_93) : "r"(_fp4_bf16x8_23[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_93;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_94;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_94) : "r"(_fp4_bf16x8_23[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_94;
    uint32_t _bf16x2_mul_95;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_95) : "r"(_fp4_bf16x8_23[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_95;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 26624 + (unsigned int)((128 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (128 + pair * 16 + a_col) % 64 * 2 ^ ((128 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (128 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 26624 + 4096 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 8192 + 2048 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_24[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_24[0]), "=r"(_fp4_bf16x8_24[1]), "=r"(_fp4_bf16x8_24[2]), "=r"(_fp4_bf16x8_24[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_24[0]), "=r"(_fp4_bf16x8_24[1]), "=r"(_fp4_bf16x8_24[2]), "=r"(_fp4_bf16x8_24[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_24[0], hi = (uint16_t)(_fp4_bf16x8_24[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_24[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_24[1], hi = (uint16_t)(_fp4_bf16x8_24[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_24[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_24[2], hi = (uint16_t)(_fp4_bf16x8_24[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_24[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_24[3], hi = (uint16_t)(_fp4_bf16x8_24[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_24[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_25[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_25[0]), "=r"(_fp4_bf16x8_25[1]), "=r"(_fp4_bf16x8_25[2]), "=r"(_fp4_bf16x8_25[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_25[0]), "=r"(_fp4_bf16x8_25[1]), "=r"(_fp4_bf16x8_25[2]), "=r"(_fp4_bf16x8_25[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_25[0], hi = (uint16_t)(_fp4_bf16x8_25[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_25[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_25[1], hi = (uint16_t)(_fp4_bf16x8_25[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_25[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_25[2], hi = (uint16_t)(_fp4_bf16x8_25[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_25[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_25[3], hi = (uint16_t)(_fp4_bf16x8_25[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_25[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_96;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_96) : "r"(_fp4_bf16x8_24[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_96;
    uint32_t _bf16x2_mul_97;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_97) : "r"(_fp4_bf16x8_24[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_97;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_98;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_98) : "r"(_fp4_bf16x8_24[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_98;
    uint32_t _bf16x2_mul_99;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_99) : "r"(_fp4_bf16x8_24[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_99;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_100;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_100) : "r"(_fp4_bf16x8_25[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_100;
    uint32_t _bf16x2_mul_101;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_101) : "r"(_fp4_bf16x8_25[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_101;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_102;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_102) : "r"(_fp4_bf16x8_25[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_102;
    uint32_t _bf16x2_mul_103;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_103) : "r"(_fp4_bf16x8_25[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_103;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 26624 + (unsigned int)((128 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (128 + (2 + pair) * 16 + a_col) % 64 * 2 ^ ((128 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (128 + (2 + pair) * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 26624 + 4096 + (unsigned int)((2 + pair) * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 8192 + 2048 + (unsigned int)((2 + pair) * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_26[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_26[0]), "=r"(_fp4_bf16x8_26[1]), "=r"(_fp4_bf16x8_26[2]), "=r"(_fp4_bf16x8_26[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_26[0]), "=r"(_fp4_bf16x8_26[1]), "=r"(_fp4_bf16x8_26[2]), "=r"(_fp4_bf16x8_26[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_26[0], hi = (uint16_t)(_fp4_bf16x8_26[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_26[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_26[1], hi = (uint16_t)(_fp4_bf16x8_26[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_26[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_26[2], hi = (uint16_t)(_fp4_bf16x8_26[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_26[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_26[3], hi = (uint16_t)(_fp4_bf16x8_26[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_26[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_27[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_27[0]), "=r"(_fp4_bf16x8_27[1]), "=r"(_fp4_bf16x8_27[2]), "=r"(_fp4_bf16x8_27[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_27[0]), "=r"(_fp4_bf16x8_27[1]), "=r"(_fp4_bf16x8_27[2]), "=r"(_fp4_bf16x8_27[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_27[0], hi = (uint16_t)(_fp4_bf16x8_27[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_27[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_27[1], hi = (uint16_t)(_fp4_bf16x8_27[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_27[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_27[2], hi = (uint16_t)(_fp4_bf16x8_27[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_27[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_27[3], hi = (uint16_t)(_fp4_bf16x8_27[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_27[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_104;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_104) : "r"(_fp4_bf16x8_26[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_104;
    uint32_t _bf16x2_mul_105;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_105) : "r"(_fp4_bf16x8_26[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_105;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_106;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_106) : "r"(_fp4_bf16x8_26[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_106;
    uint32_t _bf16x2_mul_107;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_107) : "r"(_fp4_bf16x8_26[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_107;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_108;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_108) : "r"(_fp4_bf16x8_27[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_108;
    uint32_t _bf16x2_mul_109;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_109) : "r"(_fp4_bf16x8_27[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_109;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_110;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_110) : "r"(_fp4_bf16x8_27[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_110;
    uint32_t _bf16x2_mul_111;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_111) : "r"(_fp4_bf16x8_27[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_111;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 26624 + (unsigned int)((192 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (192 + pair * 16 + a_col) % 64 * 2 ^ ((192 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (192 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 26624 + 6144 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 8192 + 3072 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_28[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_28[0]), "=r"(_fp4_bf16x8_28[1]), "=r"(_fp4_bf16x8_28[2]), "=r"(_fp4_bf16x8_28[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_28[0]), "=r"(_fp4_bf16x8_28[1]), "=r"(_fp4_bf16x8_28[2]), "=r"(_fp4_bf16x8_28[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_28[0], hi = (uint16_t)(_fp4_bf16x8_28[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_28[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_28[1], hi = (uint16_t)(_fp4_bf16x8_28[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_28[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_28[2], hi = (uint16_t)(_fp4_bf16x8_28[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_28[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_28[3], hi = (uint16_t)(_fp4_bf16x8_28[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_28[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_29[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_29[0]), "=r"(_fp4_bf16x8_29[1]), "=r"(_fp4_bf16x8_29[2]), "=r"(_fp4_bf16x8_29[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_29[0]), "=r"(_fp4_bf16x8_29[1]), "=r"(_fp4_bf16x8_29[2]), "=r"(_fp4_bf16x8_29[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_29[0], hi = (uint16_t)(_fp4_bf16x8_29[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_29[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_29[1], hi = (uint16_t)(_fp4_bf16x8_29[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_29[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_29[2], hi = (uint16_t)(_fp4_bf16x8_29[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_29[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_29[3], hi = (uint16_t)(_fp4_bf16x8_29[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_29[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_112;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_112) : "r"(_fp4_bf16x8_28[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_112;
    uint32_t _bf16x2_mul_113;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_113) : "r"(_fp4_bf16x8_28[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_113;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_114;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_114) : "r"(_fp4_bf16x8_28[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_114;
    uint32_t _bf16x2_mul_115;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_115) : "r"(_fp4_bf16x8_28[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_115;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_116;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_116) : "r"(_fp4_bf16x8_29[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_116;
    uint32_t _bf16x2_mul_117;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_117) : "r"(_fp4_bf16x8_29[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_117;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_118;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_118) : "r"(_fp4_bf16x8_29[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_118;
    uint32_t _bf16x2_mul_119;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_119) : "r"(_fp4_bf16x8_29[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_119;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 26624 + (unsigned int)((192 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (192 + (2 + pair) * 16 + a_col) % 64 * 2 ^ ((192 + (2 + pair) * 16 + a_col) / 64 * 2048 + a_row * 128 + (192 + (2 + pair) * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 26624 + 6144 + (unsigned int)((2 + pair) * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 8192 + 3072 + (unsigned int)((2 + pair) * 256) + (unsigned int)sfx_lane_off));
    uint32_t _fp4_bf16x8_30[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_30[0]), "=r"(_fp4_bf16x8_30[1]), "=r"(_fp4_bf16x8_30[2]), "=r"(_fp4_bf16x8_30[3]) : "r"((uint32_t)(raw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_30[0]), "=r"(_fp4_bf16x8_30[1]), "=r"(_fp4_bf16x8_30[2]), "=r"(_fp4_bf16x8_30[3]) : "r"((uint32_t)(raw[0])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_30[0], hi = (uint16_t)(_fp4_bf16x8_30[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_30[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_30[1], hi = (uint16_t)(_fp4_bf16x8_30[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_30[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_30[2], hi = (uint16_t)(_fp4_bf16x8_30[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_30[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_30[3], hi = (uint16_t)(_fp4_bf16x8_30[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_30[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _fp4_bf16x8_31[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_31[0]), "=r"(_fp4_bf16x8_31[1]), "=r"(_fp4_bf16x8_31[2]), "=r"(_fp4_bf16x8_31[3]) : "r"((uint32_t)(raw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_31[0]), "=r"(_fp4_bf16x8_31[1]), "=r"(_fp4_bf16x8_31[2]), "=r"(_fp4_bf16x8_31[3]) : "r"((uint32_t)(raw[1])));
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_31[0], hi = (uint16_t)(_fp4_bf16x8_31[0] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_31[0]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_31[1], hi = (uint16_t)(_fp4_bf16x8_31[1] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_31[1]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_31[2], hi = (uint16_t)(_fp4_bf16x8_31[2] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_31[2]) : "f"(f1), "f"(f0));
    }
    {
    uint16_t lo = (uint16_t)_fp4_bf16x8_31[3], hi = (uint16_t)(_fp4_bf16x8_31[3] >> 16);
    float f0, f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_31[3]) : "f"(f1), "f"(f0));
    }
    #endif
    }
    uint32_t _bf16x2_mul_120;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_120) : "r"(_fp4_bf16x8_30[0]), "r"(carriers[0]));
    b_frag[0] = _bf16x2_mul_120;
    uint32_t _bf16x2_mul_121;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_121) : "r"(_fp4_bf16x8_30[1]), "r"(carriers[0]));
    b_frag[1] = _bf16x2_mul_121;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_122;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_122) : "r"(_fp4_bf16x8_30[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_122;
    uint32_t _bf16x2_mul_123;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_123) : "r"(_fp4_bf16x8_30[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_123;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_124;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_124) : "r"(_fp4_bf16x8_31[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_124;
    uint32_t _bf16x2_mul_125;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_125) : "r"(_fp4_bf16x8_31[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_125;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_126;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_126) : "r"(_fp4_bf16x8_31[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_126;
    uint32_t _bf16x2_mul_127;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_127) : "r"(_fp4_bf16x8_31[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_127;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    float part[16];
    int rec_base = (((pair - 1) * 2 + npair) * 2 + nw) * 512 + lane;
    if (pair > 0) {
        smk_part[rec_base] = acc[0];
        smk_part[rec_base + 32] = acc[1];
        smk_part[rec_base + 64] = acc[2];
        smk_part[rec_base + 96] = acc[3];
        smk_part[rec_base + 128] = acc[4];
        smk_part[rec_base + 160] = acc[5];
        smk_part[rec_base + 192] = acc[6];
        smk_part[rec_base + 224] = acc[7];
        smk_part[rec_base + 256] = acc[8];
        smk_part[rec_base + 288] = acc[9];
        smk_part[rec_base + 320] = acc[10];
        smk_part[rec_base + 352] = acc[11];
        smk_part[rec_base + 384] = acc[12];
        smk_part[rec_base + 416] = acc[13];
        smk_part[rec_base + 448] = acc[14];
        smk_part[rec_base + 480] = acc[15];
    }
    __syncthreads();
    if (pair < 1) {
        int src_base = (npair * 2 + nw) * 512 + lane;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(smk_part_addr + (unsigned int)(src_base * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(smk_part_addr + (unsigned int)((src_base + 32) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(smk_part_addr + (unsigned int)((src_base + 64) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(smk_part_addr + (unsigned int)((src_base + 96) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(smk_part_addr + (unsigned int)((src_base + 128) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(smk_part_addr + (unsigned int)((src_base + 160) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(smk_part_addr + (unsigned int)((src_base + 192) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(smk_part_addr + (unsigned int)((src_base + 224) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[8])) : "r"(smk_part_addr + (unsigned int)((src_base + 256) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[9])) : "r"(smk_part_addr + (unsigned int)((src_base + 288) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[10])) : "r"(smk_part_addr + (unsigned int)((src_base + 320) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[11])) : "r"(smk_part_addr + (unsigned int)((src_base + 352) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[12])) : "r"(smk_part_addr + (unsigned int)((src_base + 384) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[13])) : "r"(smk_part_addr + (unsigned int)((src_base + 416) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[14])) : "r"(smk_part_addr + (unsigned int)((src_base + 448) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[15])) : "r"(smk_part_addr + (unsigned int)((src_base + 480) * 4)));
        acc[0] = acc[0] + part[0];
        acc[1] = acc[1] + part[1];
        acc[2] = acc[2] + part[2];
        acc[3] = acc[3] + part[3];
        acc[4] = acc[4] + part[4];
        acc[5] = acc[5] + part[5];
        acc[6] = acc[6] + part[6];
        acc[7] = acc[7] + part[7];
        acc[8] = acc[8] + part[8];
        acc[9] = acc[9] + part[9];
        acc[10] = acc[10] + part[10];
        acc[11] = acc[11] + part[11];
        acc[12] = acc[12] + part[12];
        acc[13] = acc[13] + part[13];
        acc[14] = acc[14] + part[14];
        acc[15] = acc[15] + part[15];
        int slice_base = (npair * 2 + nw) * 2;
        if (M > 1) {
            int rec_lane = cta_rank * 1024 + lane * 16;
            if (slice_base == cta_rank) {
                smk_mail[rec_lane / 4] = acc[0];
                smk_mail[rec_lane / 4 + 1] = acc[1];
                smk_mail[rec_lane / 4 + 2] = acc[2];
                smk_mail[rec_lane / 4 + 3] = acc[3];
                smk_mail[rec_lane / 4 + 128] = acc[4];
                smk_mail[rec_lane / 4 + 129] = acc[5];
                smk_mail[rec_lane / 4 + 130] = acc[6];
                smk_mail[rec_lane / 4 + 131] = acc[7];
            } else {
                int rs_dst = smk_mail_addr + (unsigned int)rec_lane;
                uint32_t _mapa_0;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_0) : "r"(rs_dst), "r"(slice_base));
                uint32_t _mbar_peer_0;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_0) : "r"(mail_bar_addr), "r"(slice_base));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])), "r"(_mbar_peer_0) : "memory");
                uint32_t _mbar_peer_1;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_1) : "r"(mail_bar_addr), "r"(slice_base));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + 512), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])), "r"(_mbar_peer_1) : "memory");
            }
            if (slice_base + 1 == cta_rank) {
                smk_mail[rec_lane / 4] = acc[8];
                smk_mail[rec_lane / 4 + 1] = acc[9];
                smk_mail[rec_lane / 4 + 2] = acc[10];
                smk_mail[rec_lane / 4 + 3] = acc[11];
                smk_mail[rec_lane / 4 + 128] = acc[12];
                smk_mail[rec_lane / 4 + 129] = acc[13];
                smk_mail[rec_lane / 4 + 130] = acc[14];
                smk_mail[rec_lane / 4 + 131] = acc[15];
            } else {
                int rs_dst_1 = smk_mail_addr + (unsigned int)rec_lane;
                uint32_t _mapa_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_1) : "r"(rs_dst_1), "r"(slice_base + 1));
                uint32_t _mbar_peer_2;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_2) : "r"(mail_bar_addr), "r"(slice_base + 1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])), "r"(_mbar_peer_2) : "memory");
                uint32_t _mbar_peer_3;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_3) : "r"(mail_bar_addr), "r"(slice_base + 1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 512), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])), "r"(_mbar_peer_3) : "memory");
            }
        } else {
            int row0_lane = cta_rank * 64 + lane * 16;
            if (lane < 4) {
                if (slice_base == cta_rank) {
                    smk_mail[row0_lane / 4] = acc[0];
                    smk_mail[row0_lane / 4 + 1] = acc[1];
                    smk_mail[row0_lane / 4 + 2] = acc[4];
                    smk_mail[row0_lane / 4 + 3] = acc[5];
                } else {
                    int row0_dst = smk_mail_addr + (unsigned int)row0_lane;
                    uint32_t _mapa_2;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_2) : "r"(row0_dst), "r"(slice_base));
                    uint32_t _mbar_peer_4;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_4) : "r"(mail_bar_addr), "r"(slice_base));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_2), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(_mbar_peer_4) : "memory");
                }
                if (slice_base + 1 == cta_rank) {
                    smk_mail[row0_lane / 4] = acc[8];
                    smk_mail[row0_lane / 4 + 1] = acc[9];
                    smk_mail[row0_lane / 4 + 2] = acc[12];
                    smk_mail[row0_lane / 4 + 3] = acc[13];
                } else {
                    int row0_dst_1 = smk_mail_addr + (unsigned int)row0_lane;
                    uint32_t _mapa_3;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_3) : "r"(row0_dst_1), "r"(slice_base + 1));
                    uint32_t _mbar_peer_5;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_5) : "r"(mail_bar_addr), "r"(slice_base + 1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_3), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(_mbar_peer_5) : "memory");
                }
            }
        }
        if (slice_base == cta_rank) {
            mbarrier_wait_cluster_hint(mail_bar_addr, 0, 10000000);
            if (M > 1) {
                float rs_vals[64];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(0) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(4) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(512 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[8])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(8) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(1024 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[12])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(12) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(1536 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[16])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(16) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(2048 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[20])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(20) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(2560 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[24])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(24) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(3072 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[28])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(28) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(3584 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[32])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(32) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(4096 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[36])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(36) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(4608 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[40])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(40) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(5120 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[44])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(44) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(5632 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[48])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(48) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(6144 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[52])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(52) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(6656 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[56])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(56) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(7168 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[60])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals[(60) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(7680 + lane * 16)));
                float rs_sum[8];
                rs_sum[0] = rs_vals[0];
                rs_sum[1] = rs_vals[1];
                rs_sum[2] = rs_vals[2];
                rs_sum[3] = rs_vals[3];
                rs_sum[4] = rs_vals[4];
                rs_sum[5] = rs_vals[5];
                rs_sum[6] = rs_vals[6];
                rs_sum[7] = rs_vals[7];
                rs_sum[0] = rs_sum[0] + rs_vals[8];
                rs_sum[1] = rs_sum[1] + rs_vals[9];
                rs_sum[2] = rs_sum[2] + rs_vals[10];
                rs_sum[3] = rs_sum[3] + rs_vals[11];
                rs_sum[4] = rs_sum[4] + rs_vals[12];
                rs_sum[5] = rs_sum[5] + rs_vals[13];
                rs_sum[6] = rs_sum[6] + rs_vals[14];
                rs_sum[7] = rs_sum[7] + rs_vals[15];
                rs_sum[0] = rs_sum[0] + rs_vals[16];
                rs_sum[1] = rs_sum[1] + rs_vals[17];
                rs_sum[2] = rs_sum[2] + rs_vals[18];
                rs_sum[3] = rs_sum[3] + rs_vals[19];
                rs_sum[4] = rs_sum[4] + rs_vals[20];
                rs_sum[5] = rs_sum[5] + rs_vals[21];
                rs_sum[6] = rs_sum[6] + rs_vals[22];
                rs_sum[7] = rs_sum[7] + rs_vals[23];
                rs_sum[0] = rs_sum[0] + rs_vals[24];
                rs_sum[1] = rs_sum[1] + rs_vals[25];
                rs_sum[2] = rs_sum[2] + rs_vals[26];
                rs_sum[3] = rs_sum[3] + rs_vals[27];
                rs_sum[4] = rs_sum[4] + rs_vals[28];
                rs_sum[5] = rs_sum[5] + rs_vals[29];
                rs_sum[6] = rs_sum[6] + rs_vals[30];
                rs_sum[7] = rs_sum[7] + rs_vals[31];
                rs_sum[0] = rs_sum[0] + rs_vals[32];
                rs_sum[1] = rs_sum[1] + rs_vals[33];
                rs_sum[2] = rs_sum[2] + rs_vals[34];
                rs_sum[3] = rs_sum[3] + rs_vals[35];
                rs_sum[4] = rs_sum[4] + rs_vals[36];
                rs_sum[5] = rs_sum[5] + rs_vals[37];
                rs_sum[6] = rs_sum[6] + rs_vals[38];
                rs_sum[7] = rs_sum[7] + rs_vals[39];
                rs_sum[0] = rs_sum[0] + rs_vals[40];
                rs_sum[1] = rs_sum[1] + rs_vals[41];
                rs_sum[2] = rs_sum[2] + rs_vals[42];
                rs_sum[3] = rs_sum[3] + rs_vals[43];
                rs_sum[4] = rs_sum[4] + rs_vals[44];
                rs_sum[5] = rs_sum[5] + rs_vals[45];
                rs_sum[6] = rs_sum[6] + rs_vals[46];
                rs_sum[7] = rs_sum[7] + rs_vals[47];
                rs_sum[0] = rs_sum[0] + rs_vals[48];
                rs_sum[1] = rs_sum[1] + rs_vals[49];
                rs_sum[2] = rs_sum[2] + rs_vals[50];
                rs_sum[3] = rs_sum[3] + rs_vals[51];
                rs_sum[4] = rs_sum[4] + rs_vals[52];
                rs_sum[5] = rs_sum[5] + rs_vals[53];
                rs_sum[6] = rs_sum[6] + rs_vals[54];
                rs_sum[7] = rs_sum[7] + rs_vals[55];
                rs_sum[0] = rs_sum[0] + rs_vals[56];
                rs_sum[1] = rs_sum[1] + rs_vals[57];
                rs_sum[2] = rs_sum[2] + rs_vals[58];
                rs_sum[3] = rs_sum[3] + rs_vals[59];
                rs_sum[4] = rs_sum[4] + rs_vals[60];
                rs_sum[5] = rs_sum[5] + rs_vals[61];
                rs_sum[6] = rs_sum[6] + rs_vals[62];
                rs_sum[7] = rs_sum[7] + rs_vals[63];
                float alpha_value = 1.0f;
                {
                    alpha_value = alpha[0];
                }
                int row0 = lane_g;
                int row1 = lane_g + 8;
                int col = pair_off_n + nw * 8 + lane_r * 2;
                if (row0 < M) {
                    long long base0 = (long long)row0 * (long long)N + (long long)col;
                    float _scaled_0[2];
                    const float2 _scale2_6 = {alpha_value, alpha_value};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(rs_sum)[_ls], _scale2_6);
                    unsigned int _packed_0[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 0], _scaled_0[_lp*2+1 + 0]));
                        _packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + base0)[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
                    float _scaled_1[2];
                    const float2 _scale2_7 = {alpha_value, alpha_value};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum + 4))[_ls], _scale2_7);
                    unsigned int _packed_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 0], _scaled_1[_lp*2+1 + 0]));
                        _packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base0 + 16))[0] = reinterpret_cast<unsigned*>(_packed_1)[0];
                }
                if (row1 < M) {
                    long long base1 = (long long)row1 * (long long)N + (long long)col;
                    float _scaled_2[2];
                    const float2 _scale2_8 = {alpha_value, alpha_value};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum + 2))[_ls], _scale2_8);
                    unsigned int _packed_2[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 0], _scaled_2[_lp*2+1 + 0]));
                        _packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + base1)[0] = reinterpret_cast<unsigned*>(_packed_2)[0];
                    float _scaled_3[2];
                    const float2 _scale2_9 = {alpha_value, alpha_value};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum + 6))[_ls], _scale2_9);
                    unsigned int _packed_3[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 0], _scaled_3[_lp*2+1 + 0]));
                        _packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base1 + 16))[0] = reinterpret_cast<unsigned*>(_packed_3)[0];
                }
            } else if (lane < 4) {
                float row0_vals[32];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(0) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(0 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[4])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(4) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(64 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[8])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(8) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(128 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[12])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(12) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(192 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[16])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(16) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(256 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[20])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(20) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(320 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[24])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(24) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(384 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[28])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals[(28) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(448 + lane * 16)));
                float row0_sum[8];
                row0_sum[0] = 0.0f;
                row0_sum[1] = 0.0f;
                row0_sum[2] = 0.0f;
                row0_sum[3] = 0.0f;
                row0_sum[4] = 0.0f;
                row0_sum[5] = 0.0f;
                row0_sum[6] = 0.0f;
                row0_sum[7] = 0.0f;
                row0_sum[0] = row0_vals[0];
                row0_sum[1] = row0_vals[1];
                row0_sum[4] = row0_vals[2];
                row0_sum[5] = row0_vals[3];
                row0_sum[0] = row0_sum[0] + row0_vals[4];
                row0_sum[1] = row0_sum[1] + row0_vals[5];
                row0_sum[4] = row0_sum[4] + row0_vals[6];
                row0_sum[5] = row0_sum[5] + row0_vals[7];
                row0_sum[0] = row0_sum[0] + row0_vals[8];
                row0_sum[1] = row0_sum[1] + row0_vals[9];
                row0_sum[4] = row0_sum[4] + row0_vals[10];
                row0_sum[5] = row0_sum[5] + row0_vals[11];
                row0_sum[0] = row0_sum[0] + row0_vals[12];
                row0_sum[1] = row0_sum[1] + row0_vals[13];
                row0_sum[4] = row0_sum[4] + row0_vals[14];
                row0_sum[5] = row0_sum[5] + row0_vals[15];
                row0_sum[0] = row0_sum[0] + row0_vals[16];
                row0_sum[1] = row0_sum[1] + row0_vals[17];
                row0_sum[4] = row0_sum[4] + row0_vals[18];
                row0_sum[5] = row0_sum[5] + row0_vals[19];
                row0_sum[0] = row0_sum[0] + row0_vals[20];
                row0_sum[1] = row0_sum[1] + row0_vals[21];
                row0_sum[4] = row0_sum[4] + row0_vals[22];
                row0_sum[5] = row0_sum[5] + row0_vals[23];
                row0_sum[0] = row0_sum[0] + row0_vals[24];
                row0_sum[1] = row0_sum[1] + row0_vals[25];
                row0_sum[4] = row0_sum[4] + row0_vals[26];
                row0_sum[5] = row0_sum[5] + row0_vals[27];
                row0_sum[0] = row0_sum[0] + row0_vals[28];
                row0_sum[1] = row0_sum[1] + row0_vals[29];
                row0_sum[4] = row0_sum[4] + row0_vals[30];
                row0_sum[5] = row0_sum[5] + row0_vals[31];
                float alpha_value_1 = 1.0f;
                {
                    alpha_value_1 = alpha[0];
                }
                int row0_1 = lane_g;
                int row1_1 = lane_g + 8;
                int col_1 = pair_off_n + nw * 8 + lane_r * 2;
                if (row0_1 < M) {
                    long long base0_1 = (long long)row0_1 * (long long)N + (long long)col_1;
                    float _scaled_0_1[2];
                    const float2 _scale2_10 = {alpha_value_1, alpha_value_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_0_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(row0_sum)[_ls], _scale2_10);
                    unsigned int _packed_0_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0_1[_lp*2 + 0], _scaled_0_1[_lp*2+1 + 0]));
                        _packed_0_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + base0_1)[0] = reinterpret_cast<unsigned*>(_packed_0_1)[0];
                    float _scaled_1_1[2];
                    const float2 _scale2_11 = {alpha_value_1, alpha_value_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_1_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((row0_sum + 4))[_ls], _scale2_11);
                    unsigned int _packed_1_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1_1[_lp*2 + 0], _scaled_1_1[_lp*2+1 + 0]));
                        _packed_1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base0_1 + 16))[0] = reinterpret_cast<unsigned*>(_packed_1_1)[0];
                }
                if (row1_1 < M) {
                    long long base1_1 = (long long)row1_1 * (long long)N + (long long)col_1;
                    float _scaled_2_1[2];
                    const float2 _scale2_12 = {alpha_value_1, alpha_value_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_2_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((row0_sum + 2))[_ls], _scale2_12);
                    unsigned int _packed_2_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2_1[_lp*2 + 0], _scaled_2_1[_lp*2+1 + 0]));
                        _packed_2_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + base1_1)[0] = reinterpret_cast<unsigned*>(_packed_2_1)[0];
                    float _scaled_3_1[2];
                    const float2 _scale2_13 = {alpha_value_1, alpha_value_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_3_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((row0_sum + 6))[_ls], _scale2_13);
                    unsigned int _packed_3_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3_1[_lp*2 + 0], _scaled_3_1[_lp*2+1 + 0]));
                        _packed_3_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base1_1 + 16))[0] = reinterpret_cast<unsigned*>(_packed_3_1)[0];
                }
            }
        }
        if (slice_base + 1 == cta_rank) {
            mbarrier_wait_cluster_hint(mail_bar_addr, 0, 10000000);
            if (M > 1) {
                float rs_vals_1[64];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(0) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(4) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(512 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(8) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(1024 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(12) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(1536 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(16) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(2048 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(20) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(2560 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(24) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(3072 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(28) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(3584 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(32) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(32) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(32) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(4096 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(36) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(36) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(36) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(4608 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(40) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(40) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(40) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(5120 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(44) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(44) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(44) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(5632 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(48) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(48) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(48) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(6144 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(52) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(52) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(52) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(6656 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(56) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(56) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(56) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(7168 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(60) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(60) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rs_vals_1[(60) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(7680 + lane * 16)));
                float rs_sum_1[8];
                rs_sum_1[0] = rs_vals_1[0];
                rs_sum_1[1] = rs_vals_1[1];
                rs_sum_1[2] = rs_vals_1[2];
                rs_sum_1[3] = rs_vals_1[3];
                rs_sum_1[4] = rs_vals_1[4];
                rs_sum_1[5] = rs_vals_1[5];
                rs_sum_1[6] = rs_vals_1[6];
                rs_sum_1[7] = rs_vals_1[7];
                rs_sum_1[0] = rs_sum_1[0] + rs_vals_1[8];
                rs_sum_1[1] = rs_sum_1[1] + rs_vals_1[9];
                rs_sum_1[2] = rs_sum_1[2] + rs_vals_1[10];
                rs_sum_1[3] = rs_sum_1[3] + rs_vals_1[11];
                rs_sum_1[4] = rs_sum_1[4] + rs_vals_1[12];
                rs_sum_1[5] = rs_sum_1[5] + rs_vals_1[13];
                rs_sum_1[6] = rs_sum_1[6] + rs_vals_1[14];
                rs_sum_1[7] = rs_sum_1[7] + rs_vals_1[15];
                rs_sum_1[0] = rs_sum_1[0] + rs_vals_1[16];
                rs_sum_1[1] = rs_sum_1[1] + rs_vals_1[17];
                rs_sum_1[2] = rs_sum_1[2] + rs_vals_1[18];
                rs_sum_1[3] = rs_sum_1[3] + rs_vals_1[19];
                rs_sum_1[4] = rs_sum_1[4] + rs_vals_1[20];
                rs_sum_1[5] = rs_sum_1[5] + rs_vals_1[21];
                rs_sum_1[6] = rs_sum_1[6] + rs_vals_1[22];
                rs_sum_1[7] = rs_sum_1[7] + rs_vals_1[23];
                rs_sum_1[0] = rs_sum_1[0] + rs_vals_1[24];
                rs_sum_1[1] = rs_sum_1[1] + rs_vals_1[25];
                rs_sum_1[2] = rs_sum_1[2] + rs_vals_1[26];
                rs_sum_1[3] = rs_sum_1[3] + rs_vals_1[27];
                rs_sum_1[4] = rs_sum_1[4] + rs_vals_1[28];
                rs_sum_1[5] = rs_sum_1[5] + rs_vals_1[29];
                rs_sum_1[6] = rs_sum_1[6] + rs_vals_1[30];
                rs_sum_1[7] = rs_sum_1[7] + rs_vals_1[31];
                rs_sum_1[0] = rs_sum_1[0] + rs_vals_1[32];
                rs_sum_1[1] = rs_sum_1[1] + rs_vals_1[33];
                rs_sum_1[2] = rs_sum_1[2] + rs_vals_1[34];
                rs_sum_1[3] = rs_sum_1[3] + rs_vals_1[35];
                rs_sum_1[4] = rs_sum_1[4] + rs_vals_1[36];
                rs_sum_1[5] = rs_sum_1[5] + rs_vals_1[37];
                rs_sum_1[6] = rs_sum_1[6] + rs_vals_1[38];
                rs_sum_1[7] = rs_sum_1[7] + rs_vals_1[39];
                rs_sum_1[0] = rs_sum_1[0] + rs_vals_1[40];
                rs_sum_1[1] = rs_sum_1[1] + rs_vals_1[41];
                rs_sum_1[2] = rs_sum_1[2] + rs_vals_1[42];
                rs_sum_1[3] = rs_sum_1[3] + rs_vals_1[43];
                rs_sum_1[4] = rs_sum_1[4] + rs_vals_1[44];
                rs_sum_1[5] = rs_sum_1[5] + rs_vals_1[45];
                rs_sum_1[6] = rs_sum_1[6] + rs_vals_1[46];
                rs_sum_1[7] = rs_sum_1[7] + rs_vals_1[47];
                rs_sum_1[0] = rs_sum_1[0] + rs_vals_1[48];
                rs_sum_1[1] = rs_sum_1[1] + rs_vals_1[49];
                rs_sum_1[2] = rs_sum_1[2] + rs_vals_1[50];
                rs_sum_1[3] = rs_sum_1[3] + rs_vals_1[51];
                rs_sum_1[4] = rs_sum_1[4] + rs_vals_1[52];
                rs_sum_1[5] = rs_sum_1[5] + rs_vals_1[53];
                rs_sum_1[6] = rs_sum_1[6] + rs_vals_1[54];
                rs_sum_1[7] = rs_sum_1[7] + rs_vals_1[55];
                rs_sum_1[0] = rs_sum_1[0] + rs_vals_1[56];
                rs_sum_1[1] = rs_sum_1[1] + rs_vals_1[57];
                rs_sum_1[2] = rs_sum_1[2] + rs_vals_1[58];
                rs_sum_1[3] = rs_sum_1[3] + rs_vals_1[59];
                rs_sum_1[4] = rs_sum_1[4] + rs_vals_1[60];
                rs_sum_1[5] = rs_sum_1[5] + rs_vals_1[61];
                rs_sum_1[6] = rs_sum_1[6] + rs_vals_1[62];
                rs_sum_1[7] = rs_sum_1[7] + rs_vals_1[63];
                float alpha_value_2 = 1.0f;
                {
                    alpha_value_2 = alpha[0];
                }
                int row0_2 = lane_g;
                int row1_2 = lane_g + 8;
                int col_2 = pair_off_n + nw * 8 + lane_r * 2;
                if (row0_2 < M) {
                    long long base0_2 = (long long)row0_2 * (long long)N + (long long)col_2;
                    float _scaled_4[2];
                    const float2 _scale2_14 = {alpha_value_2, alpha_value_2};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_4)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(rs_sum_1)[_ls], _scale2_14);
                    unsigned int _packed_4[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4[_lp*2 + 0], _scaled_4[_lp*2+1 + 0]));
                        _packed_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base0_2 + 32))[0] = reinterpret_cast<unsigned*>(_packed_4)[0];
                    float _scaled_5[2];
                    const float2 _scale2_15 = {alpha_value_2, alpha_value_2};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_5)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum_1 + 4))[_ls], _scale2_15);
                    unsigned int _packed_5[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 0], _scaled_5[_lp*2+1 + 0]));
                        _packed_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base0_2 + 48))[0] = reinterpret_cast<unsigned*>(_packed_5)[0];
                }
                if (row1_2 < M) {
                    long long base1_2 = (long long)row1_2 * (long long)N + (long long)col_2;
                    float _scaled_6[2];
                    const float2 _scale2_16 = {alpha_value_2, alpha_value_2};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_6)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum_1 + 2))[_ls], _scale2_16);
                    unsigned int _packed_6[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 0], _scaled_6[_lp*2+1 + 0]));
                        _packed_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base1_2 + 32))[0] = reinterpret_cast<unsigned*>(_packed_6)[0];
                    float _scaled_7[2];
                    const float2 _scale2_17 = {alpha_value_2, alpha_value_2};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_7)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum_1 + 6))[_ls], _scale2_17);
                    unsigned int _packed_7[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 0], _scaled_7[_lp*2+1 + 0]));
                        _packed_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base1_2 + 48))[0] = reinterpret_cast<unsigned*>(_packed_7)[0];
                }
            } else if (lane < 4) {
                float row0_vals_1[32];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(0) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(0 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(4) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(64 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(8) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(128 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(12) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(192 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(16) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(256 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(20) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(320 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(24) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(384 + lane * 16)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row0_vals_1[(28) + 3]))
                    : "r"(smk_mail_addr + (unsigned int)(448 + lane * 16)));
                float row0_sum_1[8];
                row0_sum_1[0] = 0.0f;
                row0_sum_1[1] = 0.0f;
                row0_sum_1[2] = 0.0f;
                row0_sum_1[3] = 0.0f;
                row0_sum_1[4] = 0.0f;
                row0_sum_1[5] = 0.0f;
                row0_sum_1[6] = 0.0f;
                row0_sum_1[7] = 0.0f;
                row0_sum_1[0] = row0_vals_1[0];
                row0_sum_1[1] = row0_vals_1[1];
                row0_sum_1[4] = row0_vals_1[2];
                row0_sum_1[5] = row0_vals_1[3];
                row0_sum_1[0] = row0_sum_1[0] + row0_vals_1[4];
                row0_sum_1[1] = row0_sum_1[1] + row0_vals_1[5];
                row0_sum_1[4] = row0_sum_1[4] + row0_vals_1[6];
                row0_sum_1[5] = row0_sum_1[5] + row0_vals_1[7];
                row0_sum_1[0] = row0_sum_1[0] + row0_vals_1[8];
                row0_sum_1[1] = row0_sum_1[1] + row0_vals_1[9];
                row0_sum_1[4] = row0_sum_1[4] + row0_vals_1[10];
                row0_sum_1[5] = row0_sum_1[5] + row0_vals_1[11];
                row0_sum_1[0] = row0_sum_1[0] + row0_vals_1[12];
                row0_sum_1[1] = row0_sum_1[1] + row0_vals_1[13];
                row0_sum_1[4] = row0_sum_1[4] + row0_vals_1[14];
                row0_sum_1[5] = row0_sum_1[5] + row0_vals_1[15];
                row0_sum_1[0] = row0_sum_1[0] + row0_vals_1[16];
                row0_sum_1[1] = row0_sum_1[1] + row0_vals_1[17];
                row0_sum_1[4] = row0_sum_1[4] + row0_vals_1[18];
                row0_sum_1[5] = row0_sum_1[5] + row0_vals_1[19];
                row0_sum_1[0] = row0_sum_1[0] + row0_vals_1[20];
                row0_sum_1[1] = row0_sum_1[1] + row0_vals_1[21];
                row0_sum_1[4] = row0_sum_1[4] + row0_vals_1[22];
                row0_sum_1[5] = row0_sum_1[5] + row0_vals_1[23];
                row0_sum_1[0] = row0_sum_1[0] + row0_vals_1[24];
                row0_sum_1[1] = row0_sum_1[1] + row0_vals_1[25];
                row0_sum_1[4] = row0_sum_1[4] + row0_vals_1[26];
                row0_sum_1[5] = row0_sum_1[5] + row0_vals_1[27];
                row0_sum_1[0] = row0_sum_1[0] + row0_vals_1[28];
                row0_sum_1[1] = row0_sum_1[1] + row0_vals_1[29];
                row0_sum_1[4] = row0_sum_1[4] + row0_vals_1[30];
                row0_sum_1[5] = row0_sum_1[5] + row0_vals_1[31];
                float alpha_value_3 = 1.0f;
                {
                    alpha_value_3 = alpha[0];
                }
                int row0_3 = lane_g;
                int row1_3 = lane_g + 8;
                int col_3 = pair_off_n + nw * 8 + lane_r * 2;
                if (row0_3 < M) {
                    long long base0_3 = (long long)row0_3 * (long long)N + (long long)col_3;
                    float _scaled_4_1[2];
                    const float2 _scale2_18 = {alpha_value_3, alpha_value_3};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_4_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(row0_sum_1)[_ls], _scale2_18);
                    unsigned int _packed_4_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4_1[_lp*2 + 0], _scaled_4_1[_lp*2+1 + 0]));
                        _packed_4_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base0_3 + 32))[0] = reinterpret_cast<unsigned*>(_packed_4_1)[0];
                    float _scaled_5_1[2];
                    const float2 _scale2_19 = {alpha_value_3, alpha_value_3};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_5_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((row0_sum_1 + 4))[_ls], _scale2_19);
                    unsigned int _packed_5_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5_1[_lp*2 + 0], _scaled_5_1[_lp*2+1 + 0]));
                        _packed_5_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base0_3 + 48))[0] = reinterpret_cast<unsigned*>(_packed_5_1)[0];
                }
                if (row1_3 < M) {
                    long long base1_3 = (long long)row1_3 * (long long)N + (long long)col_3;
                    float _scaled_6_1[2];
                    const float2 _scale2_20 = {alpha_value_3, alpha_value_3};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_6_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((row0_sum_1 + 2))[_ls], _scale2_20);
                    unsigned int _packed_6_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6_1[_lp*2 + 0], _scaled_6_1[_lp*2+1 + 0]));
                        _packed_6_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base1_3 + 32))[0] = reinterpret_cast<unsigned*>(_packed_6_1)[0];
                    float _scaled_7_1[2];
                    const float2 _scale2_21 = {alpha_value_3, alpha_value_3};
                    #pragma unroll
                    for (int _ls = 0; _ls < 1; _ls++)
                        reinterpret_cast<float2*>(_scaled_7_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((row0_sum_1 + 6))[_ls], _scale2_21);
                    unsigned int _packed_7_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7_1[_lp*2 + 0], _scaled_7_1[_lp*2+1 + 0]));
                        _packed_7_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(C + (base1_3 + 48))[0] = reinterpret_cast<unsigned*>(_packed_7_1)[0];
                }
            }
        }
    }
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
}

} // extern "C"
