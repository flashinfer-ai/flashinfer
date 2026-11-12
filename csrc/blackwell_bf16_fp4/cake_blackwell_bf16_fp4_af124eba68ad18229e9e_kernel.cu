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
#define SMEM_SMK_A_STRIDE 16384
#define SMEM_SMK_B_OFF 9216
#define SMEM_SMK_B_STAGE_BYTES 8192
#define SMEM_SMK_B_STRIDE 16384
#define SMEM_SMK_SF_OFF 82944
#define SMEM_SMK_SF_STAGE_BYTES 4096
#define SMEM_SMK_SF_STRIDE 4096
#define SMEM_SMK_PART_OFF 66560
#define SMEM_SMK_PART_STAGE_BYTES 12288
#define SMEM_SMK_PART_STRIDE 12288
#define SMEM_SMK_MAIL_OFF 78848
#define SMEM_SMK_MAIL_STAGE_BYTES 4096
#define SMEM_SMK_MAIL_STRIDE 4096
#define SMEM_TOTAL 87040
#define THREADS 256
#define HAS_ALPHA 0
#define ENABLE_PDL 1
#define LAUNCH_MIN_BLOCKS 2

__device__ __forceinline__ uint32_t cvt_bf16x2_e4m3x2(uint16_t src) {
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(dst) : "h"(src));
#else
    uint32_t _f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(src));
    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
    float _f0;
    float _f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(dst) : "f"(_f1), "f"(_f0));
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(1,1,4) void
kernel_cake_blackwell_bf16_fp4_af124eba68ad18229e9e(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
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
    #define mail_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = (((blockIdx.z / 4) * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * gridDim.y * (gridDim.z / 4);

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smk_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smk_a_addr = smem + 1024;
    uint8_t* smk_b = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smk_b_addr = smem + 9216;
    uint8_t* smk_sf = reinterpret_cast<uint8_t*>(smem_raw + 82944);
    const int smk_sf_addr = smem + 82944;
    float* smk_part = reinterpret_cast<float*>(smem_raw + 66560);
    const int smk_part_addr = smem + 66560;
    float* smk_mail = reinterpret_cast<float*>(smem_raw + 78848);
    const int smk_mail_addr = smem + 78848;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 5 barriers)
    // Mbarriers at smem_raw[0..40)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // mail: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 32, 3072);
        }
    }

    // === Task calls (declaration order) ===
    int off_n = blockIdx.x * 64;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(full_addr, 20480);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b_addr), "l"((&B)), "r"(blockIdx.z * 512), "r"(off_n),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr), "l"((&B_descale)), "r"(blockIdx.z * 64), "r"(off_n),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr + 8, 16384);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b_addr + 16384), "l"((&B)), "r"(blockIdx.z * 512 + 128), "r"(off_n),
                   "r"(full_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr + 16, 16384);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b_addr + 32768), "l"((&B)), "r"(blockIdx.z * 512 + 256), "r"(off_n),
                   "r"(full_addr + 16), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr + 24, 16384);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b_addr + 49152), "l"((&B)), "r"(blockIdx.z * 512 + 384), "r"(off_n),
                   "r"(full_addr + 24), "l"(0x12F0000000000000ULL) : "memory");
        }
    }
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    float alpha_reg = 1.0f;
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(smk_a_addr), "l"((&A)), "r"(0), "r"(0), "r"(blockIdx.z * 16),
                   "r"(full_addr), "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(smk_a_addr + 16384), "l"((&A)), "r"(0), "r"(0), "r"(blockIdx.z * 16 + 4),
                   "r"(full_addr + 8), "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(smk_a_addr + 32768), "l"((&A)), "r"(0), "r"(0), "r"(blockIdx.z * 16 + 8),
                   "r"(full_addr + 16), "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(smk_a_addr + 49152), "l"((&A)), "r"(0), "r"(0), "r"(blockIdx.z * 16 + 12),
                   "r"(full_addr + 24), "l"(0x14F0000000000000ULL) : "memory");
        }
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned int a_frag[4];
    unsigned int b_frag[2];
    unsigned int xa0[4];
    unsigned int xa1[4];
    unsigned int ya0[4];
    unsigned int ya1[4];
    unsigned int braw[8];
    unsigned int sfw[4];
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
    int lane_g = lane / 4;
    int lane_r = lane % 4;
    int row_in_tile = nw * 8 + lane_g;
    int grp = pair;
    int a_group_off = grp * 2048;
    int pair_group_off = grp * 2048;
    int a_off0 = lane_g * 128 + lane_r * 32;
    int a_lane0 = (a_off0 ^ lane_g * 16) + a_group_off;
    int a_lane1 = (a_off0 + 16 ^ lane_g * 16) + a_group_off;
    int b_lane = row_in_tile * 32 + lane_r * 8 + pair_group_off;
    int sf_lane = row_in_tile * 16 + grp * 4;
    int b_lane_w = row_in_tile * 128 + ((grp * 2 + (lane_r >> 1) ^ lane_g) << 4) + (lane_r & 1) * 8;
    int sf_sw = row_in_tile >> 1 & 3;
    int sf_lane_w0 = row_in_tile * 64 + ((sf_sw ^ 0) << 4) + grp * 4;
    int sf_lane_w1 = row_in_tile * 64 + ((sf_sw ^ 1) << 4) + grp * 4;
    int sf_lane_w2 = row_in_tile * 64 + ((sf_sw ^ 2) << 4) + grp * 4;
    int sf_lane_w3 = row_in_tile * 64 + ((sf_sw ^ 3) << 4) + grp * 4;
    unsigned int sf_sel = (unsigned int)(lane_r * 17 + 17408);
    mbarrier_wait(full_addr, 0);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa0[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 3]))
        : "r"(smk_a_addr + (unsigned int)a_lane0));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 3]))
        : "r"(smk_a_addr + (unsigned int)a_lane0 + 1024));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa1[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 3]))
        : "r"(smk_a_addr + (unsigned int)a_lane1));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 3]))
        : "r"(smk_a_addr + (unsigned int)a_lane1 + 1024));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[0])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(0) + 1]))
        : "r"(smk_b_addr + (unsigned int)b_lane_w));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w0));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[2])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(2) + 1]))
        : "r"(smk_b_addr + (unsigned int)b_lane_w + 2048));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[1])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w0 + 1024));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[4])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(4) + 1]))
        : "r"(smk_b_addr + (unsigned int)b_lane_w + 4096));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[2])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w0 + 2048));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[6])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(6) + 1]))
        : "r"(smk_b_addr + (unsigned int)b_lane_w + 6144));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[3])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w0 + 3072));
    uint32_t _prmt_0;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_0) : "r"(sfw[0]), "r"(sfw[0]), "r"(sf_sel));
    uint32_t car0 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_0);
    uint32_t _prmt_1;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_1) : "r"(sfw[1]), "r"(sfw[1]), "r"(sf_sel));
    uint32_t car1 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_1);
    uint32_t _prmt_2;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_2) : "r"(sfw[2]), "r"(sfw[2]), "r"(sf_sel));
    uint32_t car2 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_2);
    uint32_t _prmt_3;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_3) : "r"(sfw[3]), "r"(sfw[3]), "r"(sf_sel));
    uint32_t car3 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_3);
    uint32_t _fp4_bf16x8_0[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_0[0]), "=r"(_fp4_bf16x8_0[1]), "=r"(_fp4_bf16x8_0[2]), "=r"(_fp4_bf16x8_0[3]) : "r"((uint32_t)(braw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_0[0]), "=r"(_fp4_bf16x8_0[1]), "=r"(_fp4_bf16x8_0[2]), "=r"(_fp4_bf16x8_0[3]) : "r"((uint32_t)(braw[0])));
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
        : "=r"(_fp4_bf16x8_1[0]), "=r"(_fp4_bf16x8_1[1]), "=r"(_fp4_bf16x8_1[2]), "=r"(_fp4_bf16x8_1[3]) : "r"((uint32_t)(braw[2])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_1[0]), "=r"(_fp4_bf16x8_1[1]), "=r"(_fp4_bf16x8_1[2]), "=r"(_fp4_bf16x8_1[3]) : "r"((uint32_t)(braw[2])));
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
    uint32_t _fp4_bf16x8_2[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_2[0]), "=r"(_fp4_bf16x8_2[1]), "=r"(_fp4_bf16x8_2[2]), "=r"(_fp4_bf16x8_2[3]) : "r"((uint32_t)(braw[4])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_2[0]), "=r"(_fp4_bf16x8_2[1]), "=r"(_fp4_bf16x8_2[2]), "=r"(_fp4_bf16x8_2[3]) : "r"((uint32_t)(braw[4])));
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
        : "=r"(_fp4_bf16x8_3[0]), "=r"(_fp4_bf16x8_3[1]), "=r"(_fp4_bf16x8_3[2]), "=r"(_fp4_bf16x8_3[3]) : "r"((uint32_t)(braw[6])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_3[0]), "=r"(_fp4_bf16x8_3[1]), "=r"(_fp4_bf16x8_3[2]), "=r"(_fp4_bf16x8_3[3]) : "r"((uint32_t)(braw[6])));
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
    a_frag[0] = xa0[0];
    a_frag[1] = ya0[0];
    a_frag[2] = xa0[1];
    a_frag[3] = ya0[1];
    uint32_t _bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(_fp4_bf16x8_0[0]), "r"(car0));
    b_frag[0] = _bf16x2_mul_0;
    uint32_t _bf16x2_mul_1;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(_fp4_bf16x8_0[1]), "r"(car0));
    b_frag[1] = _bf16x2_mul_1;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(_fp4_bf16x8_1[0]), "r"(car1));
    b_frag[0] = _bf16x2_mul_2;
    uint32_t _bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(_fp4_bf16x8_1[1]), "r"(car1));
    b_frag[1] = _bf16x2_mul_3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(_fp4_bf16x8_2[0]), "r"(car2));
    b_frag[0] = _bf16x2_mul_4;
    uint32_t _bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(_fp4_bf16x8_2[1]), "r"(car2));
    b_frag[1] = _bf16x2_mul_5;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(_fp4_bf16x8_3[0]), "r"(car3));
    b_frag[0] = _bf16x2_mul_6;
    uint32_t _bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(_fp4_bf16x8_3[1]), "r"(car3));
    b_frag[1] = _bf16x2_mul_7;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    a_frag[0] = xa0[2];
    a_frag[1] = ya0[2];
    a_frag[2] = xa0[3];
    a_frag[3] = ya0[3];
    uint32_t _bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(_fp4_bf16x8_0[2]), "r"(car0));
    b_frag[0] = _bf16x2_mul_8;
    uint32_t _bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(_fp4_bf16x8_0[3]), "r"(car0));
    b_frag[1] = _bf16x2_mul_9;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(_fp4_bf16x8_1[2]), "r"(car1));
    b_frag[0] = _bf16x2_mul_10;
    uint32_t _bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(_fp4_bf16x8_1[3]), "r"(car1));
    b_frag[1] = _bf16x2_mul_11;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(_fp4_bf16x8_2[2]), "r"(car2));
    b_frag[0] = _bf16x2_mul_12;
    uint32_t _bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(_fp4_bf16x8_2[3]), "r"(car2));
    b_frag[1] = _bf16x2_mul_13;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(_fp4_bf16x8_3[2]), "r"(car3));
    b_frag[0] = _bf16x2_mul_14;
    uint32_t _bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(_fp4_bf16x8_3[3]), "r"(car3));
    b_frag[1] = _bf16x2_mul_15;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _fp4_bf16x8_4[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_4[0]), "=r"(_fp4_bf16x8_4[1]), "=r"(_fp4_bf16x8_4[2]), "=r"(_fp4_bf16x8_4[3]) : "r"((uint32_t)(braw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_4[0]), "=r"(_fp4_bf16x8_4[1]), "=r"(_fp4_bf16x8_4[2]), "=r"(_fp4_bf16x8_4[3]) : "r"((uint32_t)(braw[1])));
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
        : "=r"(_fp4_bf16x8_5[0]), "=r"(_fp4_bf16x8_5[1]), "=r"(_fp4_bf16x8_5[2]), "=r"(_fp4_bf16x8_5[3]) : "r"((uint32_t)(braw[3])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_5[0]), "=r"(_fp4_bf16x8_5[1]), "=r"(_fp4_bf16x8_5[2]), "=r"(_fp4_bf16x8_5[3]) : "r"((uint32_t)(braw[3])));
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
    uint32_t _fp4_bf16x8_6[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_6[0]), "=r"(_fp4_bf16x8_6[1]), "=r"(_fp4_bf16x8_6[2]), "=r"(_fp4_bf16x8_6[3]) : "r"((uint32_t)(braw[5])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_6[0]), "=r"(_fp4_bf16x8_6[1]), "=r"(_fp4_bf16x8_6[2]), "=r"(_fp4_bf16x8_6[3]) : "r"((uint32_t)(braw[5])));
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
        : "=r"(_fp4_bf16x8_7[0]), "=r"(_fp4_bf16x8_7[1]), "=r"(_fp4_bf16x8_7[2]), "=r"(_fp4_bf16x8_7[3]) : "r"((uint32_t)(braw[7])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_7[0]), "=r"(_fp4_bf16x8_7[1]), "=r"(_fp4_bf16x8_7[2]), "=r"(_fp4_bf16x8_7[3]) : "r"((uint32_t)(braw[7])));
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
    a_frag[0] = xa1[0];
    a_frag[1] = ya1[0];
    a_frag[2] = xa1[1];
    a_frag[3] = ya1[1];
    uint32_t _bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(_fp4_bf16x8_4[0]), "r"(car0));
    b_frag[0] = _bf16x2_mul_16;
    uint32_t _bf16x2_mul_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(_fp4_bf16x8_4[1]), "r"(car0));
    b_frag[1] = _bf16x2_mul_17;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(_fp4_bf16x8_5[0]), "r"(car1));
    b_frag[0] = _bf16x2_mul_18;
    uint32_t _bf16x2_mul_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(_fp4_bf16x8_5[1]), "r"(car1));
    b_frag[1] = _bf16x2_mul_19;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(_fp4_bf16x8_6[0]), "r"(car2));
    b_frag[0] = _bf16x2_mul_20;
    uint32_t _bf16x2_mul_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(_fp4_bf16x8_6[1]), "r"(car2));
    b_frag[1] = _bf16x2_mul_21;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(_fp4_bf16x8_7[0]), "r"(car3));
    b_frag[0] = _bf16x2_mul_22;
    uint32_t _bf16x2_mul_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(_fp4_bf16x8_7[1]), "r"(car3));
    b_frag[1] = _bf16x2_mul_23;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    a_frag[0] = xa1[2];
    a_frag[1] = ya1[2];
    a_frag[2] = xa1[3];
    a_frag[3] = ya1[3];
    uint32_t _bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(_fp4_bf16x8_4[2]), "r"(car0));
    b_frag[0] = _bf16x2_mul_24;
    uint32_t _bf16x2_mul_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(_fp4_bf16x8_4[3]), "r"(car0));
    b_frag[1] = _bf16x2_mul_25;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(_fp4_bf16x8_5[2]), "r"(car1));
    b_frag[0] = _bf16x2_mul_26;
    uint32_t _bf16x2_mul_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(_fp4_bf16x8_5[3]), "r"(car1));
    b_frag[1] = _bf16x2_mul_27;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(_fp4_bf16x8_6[2]), "r"(car2));
    b_frag[0] = _bf16x2_mul_28;
    uint32_t _bf16x2_mul_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(_fp4_bf16x8_6[3]), "r"(car2));
    b_frag[1] = _bf16x2_mul_29;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(_fp4_bf16x8_7[2]), "r"(car3));
    b_frag[0] = _bf16x2_mul_30;
    uint32_t _bf16x2_mul_31;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(_fp4_bf16x8_7[3]), "r"(car3));
    b_frag[1] = _bf16x2_mul_31;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    mbarrier_wait(full_addr + 8, 0);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa0[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 3]))
        : "r"(smk_a_addr + 16384 + (unsigned int)a_lane0));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 3]))
        : "r"(smk_a_addr + 16384 + (unsigned int)a_lane0 + 1024));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa1[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 3]))
        : "r"(smk_a_addr + 16384 + (unsigned int)a_lane1));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 3]))
        : "r"(smk_a_addr + 16384 + (unsigned int)a_lane1 + 1024));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[0])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(0) + 1]))
        : "r"(smk_b_addr + 16384 + (unsigned int)b_lane_w));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w1));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[2])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(2) + 1]))
        : "r"(smk_b_addr + 16384 + (unsigned int)b_lane_w + 2048));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[1])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w1 + 1024));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[4])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(4) + 1]))
        : "r"(smk_b_addr + 16384 + (unsigned int)b_lane_w + 4096));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[2])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w1 + 2048));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[6])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(6) + 1]))
        : "r"(smk_b_addr + 16384 + (unsigned int)b_lane_w + 6144));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[3])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w1 + 3072));
    uint32_t _prmt_4;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_4) : "r"(sfw[0]), "r"(sfw[0]), "r"(sf_sel));
    uint32_t car0_1 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_4);
    uint32_t _prmt_5;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_5) : "r"(sfw[1]), "r"(sfw[1]), "r"(sf_sel));
    uint32_t car1_2 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_5);
    uint32_t _prmt_6;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_6) : "r"(sfw[2]), "r"(sfw[2]), "r"(sf_sel));
    uint32_t car2_3 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_6);
    uint32_t _prmt_7;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_7) : "r"(sfw[3]), "r"(sfw[3]), "r"(sf_sel));
    uint32_t car3_4 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_7);
    uint32_t _fp4_bf16x8_8[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_8[0]), "=r"(_fp4_bf16x8_8[1]), "=r"(_fp4_bf16x8_8[2]), "=r"(_fp4_bf16x8_8[3]) : "r"((uint32_t)(braw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_8[0]), "=r"(_fp4_bf16x8_8[1]), "=r"(_fp4_bf16x8_8[2]), "=r"(_fp4_bf16x8_8[3]) : "r"((uint32_t)(braw[0])));
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
        : "=r"(_fp4_bf16x8_9[0]), "=r"(_fp4_bf16x8_9[1]), "=r"(_fp4_bf16x8_9[2]), "=r"(_fp4_bf16x8_9[3]) : "r"((uint32_t)(braw[2])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_9[0]), "=r"(_fp4_bf16x8_9[1]), "=r"(_fp4_bf16x8_9[2]), "=r"(_fp4_bf16x8_9[3]) : "r"((uint32_t)(braw[2])));
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
    uint32_t _fp4_bf16x8_10[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_10[0]), "=r"(_fp4_bf16x8_10[1]), "=r"(_fp4_bf16x8_10[2]), "=r"(_fp4_bf16x8_10[3]) : "r"((uint32_t)(braw[4])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_10[0]), "=r"(_fp4_bf16x8_10[1]), "=r"(_fp4_bf16x8_10[2]), "=r"(_fp4_bf16x8_10[3]) : "r"((uint32_t)(braw[4])));
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
        : "=r"(_fp4_bf16x8_11[0]), "=r"(_fp4_bf16x8_11[1]), "=r"(_fp4_bf16x8_11[2]), "=r"(_fp4_bf16x8_11[3]) : "r"((uint32_t)(braw[6])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_11[0]), "=r"(_fp4_bf16x8_11[1]), "=r"(_fp4_bf16x8_11[2]), "=r"(_fp4_bf16x8_11[3]) : "r"((uint32_t)(braw[6])));
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
    a_frag[0] = xa0[0];
    a_frag[1] = ya0[0];
    a_frag[2] = xa0[1];
    a_frag[3] = ya0[1];
    uint32_t _bf16x2_mul_32;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(_fp4_bf16x8_8[0]), "r"(car0_1));
    b_frag[0] = _bf16x2_mul_32;
    uint32_t _bf16x2_mul_33;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(_fp4_bf16x8_8[1]), "r"(car0_1));
    b_frag[1] = _bf16x2_mul_33;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_34;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(_fp4_bf16x8_9[0]), "r"(car1_2));
    b_frag[0] = _bf16x2_mul_34;
    uint32_t _bf16x2_mul_35;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(_fp4_bf16x8_9[1]), "r"(car1_2));
    b_frag[1] = _bf16x2_mul_35;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_36;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(_fp4_bf16x8_10[0]), "r"(car2_3));
    b_frag[0] = _bf16x2_mul_36;
    uint32_t _bf16x2_mul_37;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(_fp4_bf16x8_10[1]), "r"(car2_3));
    b_frag[1] = _bf16x2_mul_37;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_38;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(_fp4_bf16x8_11[0]), "r"(car3_4));
    b_frag[0] = _bf16x2_mul_38;
    uint32_t _bf16x2_mul_39;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(_fp4_bf16x8_11[1]), "r"(car3_4));
    b_frag[1] = _bf16x2_mul_39;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    a_frag[0] = xa0[2];
    a_frag[1] = ya0[2];
    a_frag[2] = xa0[3];
    a_frag[3] = ya0[3];
    uint32_t _bf16x2_mul_40;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(_fp4_bf16x8_8[2]), "r"(car0_1));
    b_frag[0] = _bf16x2_mul_40;
    uint32_t _bf16x2_mul_41;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(_fp4_bf16x8_8[3]), "r"(car0_1));
    b_frag[1] = _bf16x2_mul_41;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_42;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(_fp4_bf16x8_9[2]), "r"(car1_2));
    b_frag[0] = _bf16x2_mul_42;
    uint32_t _bf16x2_mul_43;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(_fp4_bf16x8_9[3]), "r"(car1_2));
    b_frag[1] = _bf16x2_mul_43;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_44;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(_fp4_bf16x8_10[2]), "r"(car2_3));
    b_frag[0] = _bf16x2_mul_44;
    uint32_t _bf16x2_mul_45;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(_fp4_bf16x8_10[3]), "r"(car2_3));
    b_frag[1] = _bf16x2_mul_45;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_46;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(_fp4_bf16x8_11[2]), "r"(car3_4));
    b_frag[0] = _bf16x2_mul_46;
    uint32_t _bf16x2_mul_47;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(_fp4_bf16x8_11[3]), "r"(car3_4));
    b_frag[1] = _bf16x2_mul_47;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _fp4_bf16x8_12[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_12[0]), "=r"(_fp4_bf16x8_12[1]), "=r"(_fp4_bf16x8_12[2]), "=r"(_fp4_bf16x8_12[3]) : "r"((uint32_t)(braw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_12[0]), "=r"(_fp4_bf16x8_12[1]), "=r"(_fp4_bf16x8_12[2]), "=r"(_fp4_bf16x8_12[3]) : "r"((uint32_t)(braw[1])));
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
        : "=r"(_fp4_bf16x8_13[0]), "=r"(_fp4_bf16x8_13[1]), "=r"(_fp4_bf16x8_13[2]), "=r"(_fp4_bf16x8_13[3]) : "r"((uint32_t)(braw[3])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_13[0]), "=r"(_fp4_bf16x8_13[1]), "=r"(_fp4_bf16x8_13[2]), "=r"(_fp4_bf16x8_13[3]) : "r"((uint32_t)(braw[3])));
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
    uint32_t _fp4_bf16x8_14[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_14[0]), "=r"(_fp4_bf16x8_14[1]), "=r"(_fp4_bf16x8_14[2]), "=r"(_fp4_bf16x8_14[3]) : "r"((uint32_t)(braw[5])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_14[0]), "=r"(_fp4_bf16x8_14[1]), "=r"(_fp4_bf16x8_14[2]), "=r"(_fp4_bf16x8_14[3]) : "r"((uint32_t)(braw[5])));
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
        : "=r"(_fp4_bf16x8_15[0]), "=r"(_fp4_bf16x8_15[1]), "=r"(_fp4_bf16x8_15[2]), "=r"(_fp4_bf16x8_15[3]) : "r"((uint32_t)(braw[7])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_15[0]), "=r"(_fp4_bf16x8_15[1]), "=r"(_fp4_bf16x8_15[2]), "=r"(_fp4_bf16x8_15[3]) : "r"((uint32_t)(braw[7])));
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
    a_frag[0] = xa1[0];
    a_frag[1] = ya1[0];
    a_frag[2] = xa1[1];
    a_frag[3] = ya1[1];
    uint32_t _bf16x2_mul_48;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(_fp4_bf16x8_12[0]), "r"(car0_1));
    b_frag[0] = _bf16x2_mul_48;
    uint32_t _bf16x2_mul_49;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(_fp4_bf16x8_12[1]), "r"(car0_1));
    b_frag[1] = _bf16x2_mul_49;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_50;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(_fp4_bf16x8_13[0]), "r"(car1_2));
    b_frag[0] = _bf16x2_mul_50;
    uint32_t _bf16x2_mul_51;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(_fp4_bf16x8_13[1]), "r"(car1_2));
    b_frag[1] = _bf16x2_mul_51;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_52;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(_fp4_bf16x8_14[0]), "r"(car2_3));
    b_frag[0] = _bf16x2_mul_52;
    uint32_t _bf16x2_mul_53;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(_fp4_bf16x8_14[1]), "r"(car2_3));
    b_frag[1] = _bf16x2_mul_53;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_54;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(_fp4_bf16x8_15[0]), "r"(car3_4));
    b_frag[0] = _bf16x2_mul_54;
    uint32_t _bf16x2_mul_55;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(_fp4_bf16x8_15[1]), "r"(car3_4));
    b_frag[1] = _bf16x2_mul_55;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    a_frag[0] = xa1[2];
    a_frag[1] = ya1[2];
    a_frag[2] = xa1[3];
    a_frag[3] = ya1[3];
    uint32_t _bf16x2_mul_56;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(_fp4_bf16x8_12[2]), "r"(car0_1));
    b_frag[0] = _bf16x2_mul_56;
    uint32_t _bf16x2_mul_57;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(_fp4_bf16x8_12[3]), "r"(car0_1));
    b_frag[1] = _bf16x2_mul_57;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_58;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(_fp4_bf16x8_13[2]), "r"(car1_2));
    b_frag[0] = _bf16x2_mul_58;
    uint32_t _bf16x2_mul_59;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(_fp4_bf16x8_13[3]), "r"(car1_2));
    b_frag[1] = _bf16x2_mul_59;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_60;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(_fp4_bf16x8_14[2]), "r"(car2_3));
    b_frag[0] = _bf16x2_mul_60;
    uint32_t _bf16x2_mul_61;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(_fp4_bf16x8_14[3]), "r"(car2_3));
    b_frag[1] = _bf16x2_mul_61;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_62;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(_fp4_bf16x8_15[2]), "r"(car3_4));
    b_frag[0] = _bf16x2_mul_62;
    uint32_t _bf16x2_mul_63;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(_fp4_bf16x8_15[3]), "r"(car3_4));
    b_frag[1] = _bf16x2_mul_63;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    mbarrier_wait(full_addr + 16, 0);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa0[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 3]))
        : "r"(smk_a_addr + 32768 + (unsigned int)a_lane0));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 3]))
        : "r"(smk_a_addr + 32768 + (unsigned int)a_lane0 + 1024));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa1[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 3]))
        : "r"(smk_a_addr + 32768 + (unsigned int)a_lane1));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 3]))
        : "r"(smk_a_addr + 32768 + (unsigned int)a_lane1 + 1024));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[0])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(0) + 1]))
        : "r"(smk_b_addr + 32768 + (unsigned int)b_lane_w));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w2));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[2])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(2) + 1]))
        : "r"(smk_b_addr + 32768 + (unsigned int)b_lane_w + 2048));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[1])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w2 + 1024));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[4])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(4) + 1]))
        : "r"(smk_b_addr + 32768 + (unsigned int)b_lane_w + 4096));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[2])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w2 + 2048));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[6])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(6) + 1]))
        : "r"(smk_b_addr + 32768 + (unsigned int)b_lane_w + 6144));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[3])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w2 + 3072));
    uint32_t _prmt_8;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_8) : "r"(sfw[0]), "r"(sfw[0]), "r"(sf_sel));
    uint32_t car0_5 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_8);
    uint32_t _prmt_9;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_9) : "r"(sfw[1]), "r"(sfw[1]), "r"(sf_sel));
    uint32_t car1_6 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_9);
    uint32_t _prmt_10;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_10) : "r"(sfw[2]), "r"(sfw[2]), "r"(sf_sel));
    uint32_t car2_7 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_10);
    uint32_t _prmt_11;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_11) : "r"(sfw[3]), "r"(sfw[3]), "r"(sf_sel));
    uint32_t car3_8 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_11);
    uint32_t _fp4_bf16x8_16[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_16[0]), "=r"(_fp4_bf16x8_16[1]), "=r"(_fp4_bf16x8_16[2]), "=r"(_fp4_bf16x8_16[3]) : "r"((uint32_t)(braw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_16[0]), "=r"(_fp4_bf16x8_16[1]), "=r"(_fp4_bf16x8_16[2]), "=r"(_fp4_bf16x8_16[3]) : "r"((uint32_t)(braw[0])));
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
        : "=r"(_fp4_bf16x8_17[0]), "=r"(_fp4_bf16x8_17[1]), "=r"(_fp4_bf16x8_17[2]), "=r"(_fp4_bf16x8_17[3]) : "r"((uint32_t)(braw[2])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_17[0]), "=r"(_fp4_bf16x8_17[1]), "=r"(_fp4_bf16x8_17[2]), "=r"(_fp4_bf16x8_17[3]) : "r"((uint32_t)(braw[2])));
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
    uint32_t _fp4_bf16x8_18[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_18[0]), "=r"(_fp4_bf16x8_18[1]), "=r"(_fp4_bf16x8_18[2]), "=r"(_fp4_bf16x8_18[3]) : "r"((uint32_t)(braw[4])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_18[0]), "=r"(_fp4_bf16x8_18[1]), "=r"(_fp4_bf16x8_18[2]), "=r"(_fp4_bf16x8_18[3]) : "r"((uint32_t)(braw[4])));
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
        : "=r"(_fp4_bf16x8_19[0]), "=r"(_fp4_bf16x8_19[1]), "=r"(_fp4_bf16x8_19[2]), "=r"(_fp4_bf16x8_19[3]) : "r"((uint32_t)(braw[6])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_19[0]), "=r"(_fp4_bf16x8_19[1]), "=r"(_fp4_bf16x8_19[2]), "=r"(_fp4_bf16x8_19[3]) : "r"((uint32_t)(braw[6])));
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
    a_frag[0] = xa0[0];
    a_frag[1] = ya0[0];
    a_frag[2] = xa0[1];
    a_frag[3] = ya0[1];
    uint32_t _bf16x2_mul_64;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_64) : "r"(_fp4_bf16x8_16[0]), "r"(car0_5));
    b_frag[0] = _bf16x2_mul_64;
    uint32_t _bf16x2_mul_65;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_65) : "r"(_fp4_bf16x8_16[1]), "r"(car0_5));
    b_frag[1] = _bf16x2_mul_65;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_66;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_66) : "r"(_fp4_bf16x8_17[0]), "r"(car1_6));
    b_frag[0] = _bf16x2_mul_66;
    uint32_t _bf16x2_mul_67;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_67) : "r"(_fp4_bf16x8_17[1]), "r"(car1_6));
    b_frag[1] = _bf16x2_mul_67;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_68;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_68) : "r"(_fp4_bf16x8_18[0]), "r"(car2_7));
    b_frag[0] = _bf16x2_mul_68;
    uint32_t _bf16x2_mul_69;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_69) : "r"(_fp4_bf16x8_18[1]), "r"(car2_7));
    b_frag[1] = _bf16x2_mul_69;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_70;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_70) : "r"(_fp4_bf16x8_19[0]), "r"(car3_8));
    b_frag[0] = _bf16x2_mul_70;
    uint32_t _bf16x2_mul_71;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_71) : "r"(_fp4_bf16x8_19[1]), "r"(car3_8));
    b_frag[1] = _bf16x2_mul_71;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    a_frag[0] = xa0[2];
    a_frag[1] = ya0[2];
    a_frag[2] = xa0[3];
    a_frag[3] = ya0[3];
    uint32_t _bf16x2_mul_72;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_72) : "r"(_fp4_bf16x8_16[2]), "r"(car0_5));
    b_frag[0] = _bf16x2_mul_72;
    uint32_t _bf16x2_mul_73;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_73) : "r"(_fp4_bf16x8_16[3]), "r"(car0_5));
    b_frag[1] = _bf16x2_mul_73;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_74;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_74) : "r"(_fp4_bf16x8_17[2]), "r"(car1_6));
    b_frag[0] = _bf16x2_mul_74;
    uint32_t _bf16x2_mul_75;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_75) : "r"(_fp4_bf16x8_17[3]), "r"(car1_6));
    b_frag[1] = _bf16x2_mul_75;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_76;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_76) : "r"(_fp4_bf16x8_18[2]), "r"(car2_7));
    b_frag[0] = _bf16x2_mul_76;
    uint32_t _bf16x2_mul_77;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_77) : "r"(_fp4_bf16x8_18[3]), "r"(car2_7));
    b_frag[1] = _bf16x2_mul_77;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_78;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_78) : "r"(_fp4_bf16x8_19[2]), "r"(car3_8));
    b_frag[0] = _bf16x2_mul_78;
    uint32_t _bf16x2_mul_79;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_79) : "r"(_fp4_bf16x8_19[3]), "r"(car3_8));
    b_frag[1] = _bf16x2_mul_79;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _fp4_bf16x8_20[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_20[0]), "=r"(_fp4_bf16x8_20[1]), "=r"(_fp4_bf16x8_20[2]), "=r"(_fp4_bf16x8_20[3]) : "r"((uint32_t)(braw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_20[0]), "=r"(_fp4_bf16x8_20[1]), "=r"(_fp4_bf16x8_20[2]), "=r"(_fp4_bf16x8_20[3]) : "r"((uint32_t)(braw[1])));
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
        : "=r"(_fp4_bf16x8_21[0]), "=r"(_fp4_bf16x8_21[1]), "=r"(_fp4_bf16x8_21[2]), "=r"(_fp4_bf16x8_21[3]) : "r"((uint32_t)(braw[3])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_21[0]), "=r"(_fp4_bf16x8_21[1]), "=r"(_fp4_bf16x8_21[2]), "=r"(_fp4_bf16x8_21[3]) : "r"((uint32_t)(braw[3])));
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
    uint32_t _fp4_bf16x8_22[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_22[0]), "=r"(_fp4_bf16x8_22[1]), "=r"(_fp4_bf16x8_22[2]), "=r"(_fp4_bf16x8_22[3]) : "r"((uint32_t)(braw[5])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_22[0]), "=r"(_fp4_bf16x8_22[1]), "=r"(_fp4_bf16x8_22[2]), "=r"(_fp4_bf16x8_22[3]) : "r"((uint32_t)(braw[5])));
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
        : "=r"(_fp4_bf16x8_23[0]), "=r"(_fp4_bf16x8_23[1]), "=r"(_fp4_bf16x8_23[2]), "=r"(_fp4_bf16x8_23[3]) : "r"((uint32_t)(braw[7])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_23[0]), "=r"(_fp4_bf16x8_23[1]), "=r"(_fp4_bf16x8_23[2]), "=r"(_fp4_bf16x8_23[3]) : "r"((uint32_t)(braw[7])));
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
    a_frag[0] = xa1[0];
    a_frag[1] = ya1[0];
    a_frag[2] = xa1[1];
    a_frag[3] = ya1[1];
    uint32_t _bf16x2_mul_80;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_80) : "r"(_fp4_bf16x8_20[0]), "r"(car0_5));
    b_frag[0] = _bf16x2_mul_80;
    uint32_t _bf16x2_mul_81;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_81) : "r"(_fp4_bf16x8_20[1]), "r"(car0_5));
    b_frag[1] = _bf16x2_mul_81;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_82;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_82) : "r"(_fp4_bf16x8_21[0]), "r"(car1_6));
    b_frag[0] = _bf16x2_mul_82;
    uint32_t _bf16x2_mul_83;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_83) : "r"(_fp4_bf16x8_21[1]), "r"(car1_6));
    b_frag[1] = _bf16x2_mul_83;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_84;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_84) : "r"(_fp4_bf16x8_22[0]), "r"(car2_7));
    b_frag[0] = _bf16x2_mul_84;
    uint32_t _bf16x2_mul_85;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_85) : "r"(_fp4_bf16x8_22[1]), "r"(car2_7));
    b_frag[1] = _bf16x2_mul_85;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_86;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_86) : "r"(_fp4_bf16x8_23[0]), "r"(car3_8));
    b_frag[0] = _bf16x2_mul_86;
    uint32_t _bf16x2_mul_87;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_87) : "r"(_fp4_bf16x8_23[1]), "r"(car3_8));
    b_frag[1] = _bf16x2_mul_87;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    a_frag[0] = xa1[2];
    a_frag[1] = ya1[2];
    a_frag[2] = xa1[3];
    a_frag[3] = ya1[3];
    uint32_t _bf16x2_mul_88;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_88) : "r"(_fp4_bf16x8_20[2]), "r"(car0_5));
    b_frag[0] = _bf16x2_mul_88;
    uint32_t _bf16x2_mul_89;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_89) : "r"(_fp4_bf16x8_20[3]), "r"(car0_5));
    b_frag[1] = _bf16x2_mul_89;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_90;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_90) : "r"(_fp4_bf16x8_21[2]), "r"(car1_6));
    b_frag[0] = _bf16x2_mul_90;
    uint32_t _bf16x2_mul_91;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_91) : "r"(_fp4_bf16x8_21[3]), "r"(car1_6));
    b_frag[1] = _bf16x2_mul_91;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_92;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_92) : "r"(_fp4_bf16x8_22[2]), "r"(car2_7));
    b_frag[0] = _bf16x2_mul_92;
    uint32_t _bf16x2_mul_93;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_93) : "r"(_fp4_bf16x8_22[3]), "r"(car2_7));
    b_frag[1] = _bf16x2_mul_93;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_94;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_94) : "r"(_fp4_bf16x8_23[2]), "r"(car3_8));
    b_frag[0] = _bf16x2_mul_94;
    uint32_t _bf16x2_mul_95;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_95) : "r"(_fp4_bf16x8_23[3]), "r"(car3_8));
    b_frag[1] = _bf16x2_mul_95;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    mbarrier_wait(full_addr + 24, 0);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa0[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 3]))
        : "r"(smk_a_addr + 49152 + (unsigned int)a_lane0));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 3]))
        : "r"(smk_a_addr + 49152 + (unsigned int)a_lane0 + 1024));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa1[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 3]))
        : "r"(smk_a_addr + 49152 + (unsigned int)a_lane1));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 3]))
        : "r"(smk_a_addr + 49152 + (unsigned int)a_lane1 + 1024));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[0])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(0) + 1]))
        : "r"(smk_b_addr + 49152 + (unsigned int)b_lane_w));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w3));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[2])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(2) + 1]))
        : "r"(smk_b_addr + 49152 + (unsigned int)b_lane_w + 2048));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[1])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w3 + 1024));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[4])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(4) + 1]))
        : "r"(smk_b_addr + 49152 + (unsigned int)b_lane_w + 4096));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[2])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w3 + 2048));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[6])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(6) + 1]))
        : "r"(smk_b_addr + 49152 + (unsigned int)b_lane_w + 6144));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[3])) : "r"(smk_sf_addr + (unsigned int)sf_lane_w3 + 3072));
    uint32_t _prmt_12;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_12) : "r"(sfw[0]), "r"(sfw[0]), "r"(sf_sel));
    uint32_t car0_9 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_12);
    uint32_t _prmt_13;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_13) : "r"(sfw[1]), "r"(sfw[1]), "r"(sf_sel));
    uint32_t car1_10 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_13);
    uint32_t _prmt_14;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_14) : "r"(sfw[2]), "r"(sfw[2]), "r"(sf_sel));
    uint32_t car2_11 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_14);
    uint32_t _prmt_15;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_15) : "r"(sfw[3]), "r"(sfw[3]), "r"(sf_sel));
    uint32_t car3_12 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_15);
    uint32_t _fp4_bf16x8_24[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_24[0]), "=r"(_fp4_bf16x8_24[1]), "=r"(_fp4_bf16x8_24[2]), "=r"(_fp4_bf16x8_24[3]) : "r"((uint32_t)(braw[0])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_24[0]), "=r"(_fp4_bf16x8_24[1]), "=r"(_fp4_bf16x8_24[2]), "=r"(_fp4_bf16x8_24[3]) : "r"((uint32_t)(braw[0])));
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
        : "=r"(_fp4_bf16x8_25[0]), "=r"(_fp4_bf16x8_25[1]), "=r"(_fp4_bf16x8_25[2]), "=r"(_fp4_bf16x8_25[3]) : "r"((uint32_t)(braw[2])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_25[0]), "=r"(_fp4_bf16x8_25[1]), "=r"(_fp4_bf16x8_25[2]), "=r"(_fp4_bf16x8_25[3]) : "r"((uint32_t)(braw[2])));
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
    uint32_t _fp4_bf16x8_26[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_26[0]), "=r"(_fp4_bf16x8_26[1]), "=r"(_fp4_bf16x8_26[2]), "=r"(_fp4_bf16x8_26[3]) : "r"((uint32_t)(braw[4])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_26[0]), "=r"(_fp4_bf16x8_26[1]), "=r"(_fp4_bf16x8_26[2]), "=r"(_fp4_bf16x8_26[3]) : "r"((uint32_t)(braw[4])));
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
        : "=r"(_fp4_bf16x8_27[0]), "=r"(_fp4_bf16x8_27[1]), "=r"(_fp4_bf16x8_27[2]), "=r"(_fp4_bf16x8_27[3]) : "r"((uint32_t)(braw[6])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_27[0]), "=r"(_fp4_bf16x8_27[1]), "=r"(_fp4_bf16x8_27[2]), "=r"(_fp4_bf16x8_27[3]) : "r"((uint32_t)(braw[6])));
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
    a_frag[0] = xa0[0];
    a_frag[1] = ya0[0];
    a_frag[2] = xa0[1];
    a_frag[3] = ya0[1];
    uint32_t _bf16x2_mul_96;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_96) : "r"(_fp4_bf16x8_24[0]), "r"(car0_9));
    b_frag[0] = _bf16x2_mul_96;
    uint32_t _bf16x2_mul_97;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_97) : "r"(_fp4_bf16x8_24[1]), "r"(car0_9));
    b_frag[1] = _bf16x2_mul_97;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_98;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_98) : "r"(_fp4_bf16x8_25[0]), "r"(car1_10));
    b_frag[0] = _bf16x2_mul_98;
    uint32_t _bf16x2_mul_99;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_99) : "r"(_fp4_bf16x8_25[1]), "r"(car1_10));
    b_frag[1] = _bf16x2_mul_99;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_100;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_100) : "r"(_fp4_bf16x8_26[0]), "r"(car2_11));
    b_frag[0] = _bf16x2_mul_100;
    uint32_t _bf16x2_mul_101;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_101) : "r"(_fp4_bf16x8_26[1]), "r"(car2_11));
    b_frag[1] = _bf16x2_mul_101;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_102;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_102) : "r"(_fp4_bf16x8_27[0]), "r"(car3_12));
    b_frag[0] = _bf16x2_mul_102;
    uint32_t _bf16x2_mul_103;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_103) : "r"(_fp4_bf16x8_27[1]), "r"(car3_12));
    b_frag[1] = _bf16x2_mul_103;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    a_frag[0] = xa0[2];
    a_frag[1] = ya0[2];
    a_frag[2] = xa0[3];
    a_frag[3] = ya0[3];
    uint32_t _bf16x2_mul_104;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_104) : "r"(_fp4_bf16x8_24[2]), "r"(car0_9));
    b_frag[0] = _bf16x2_mul_104;
    uint32_t _bf16x2_mul_105;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_105) : "r"(_fp4_bf16x8_24[3]), "r"(car0_9));
    b_frag[1] = _bf16x2_mul_105;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_106;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_106) : "r"(_fp4_bf16x8_25[2]), "r"(car1_10));
    b_frag[0] = _bf16x2_mul_106;
    uint32_t _bf16x2_mul_107;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_107) : "r"(_fp4_bf16x8_25[3]), "r"(car1_10));
    b_frag[1] = _bf16x2_mul_107;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_108;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_108) : "r"(_fp4_bf16x8_26[2]), "r"(car2_11));
    b_frag[0] = _bf16x2_mul_108;
    uint32_t _bf16x2_mul_109;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_109) : "r"(_fp4_bf16x8_26[3]), "r"(car2_11));
    b_frag[1] = _bf16x2_mul_109;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_110;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_110) : "r"(_fp4_bf16x8_27[2]), "r"(car3_12));
    b_frag[0] = _bf16x2_mul_110;
    uint32_t _bf16x2_mul_111;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_111) : "r"(_fp4_bf16x8_27[3]), "r"(car3_12));
    b_frag[1] = _bf16x2_mul_111;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _fp4_bf16x8_28[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_28[0]), "=r"(_fp4_bf16x8_28[1]), "=r"(_fp4_bf16x8_28[2]), "=r"(_fp4_bf16x8_28[3]) : "r"((uint32_t)(braw[1])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_28[0]), "=r"(_fp4_bf16x8_28[1]), "=r"(_fp4_bf16x8_28[2]), "=r"(_fp4_bf16x8_28[3]) : "r"((uint32_t)(braw[1])));
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
        : "=r"(_fp4_bf16x8_29[0]), "=r"(_fp4_bf16x8_29[1]), "=r"(_fp4_bf16x8_29[2]), "=r"(_fp4_bf16x8_29[3]) : "r"((uint32_t)(braw[3])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_29[0]), "=r"(_fp4_bf16x8_29[1]), "=r"(_fp4_bf16x8_29[2]), "=r"(_fp4_bf16x8_29[3]) : "r"((uint32_t)(braw[3])));
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
    uint32_t _fp4_bf16x8_30[4];
    {
    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_30[0]), "=r"(_fp4_bf16x8_30[1]), "=r"(_fp4_bf16x8_30[2]), "=r"(_fp4_bf16x8_30[3]) : "r"((uint32_t)(braw[5])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_30[0]), "=r"(_fp4_bf16x8_30[1]), "=r"(_fp4_bf16x8_30[2]), "=r"(_fp4_bf16x8_30[3]) : "r"((uint32_t)(braw[5])));
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
        : "=r"(_fp4_bf16x8_31[0]), "=r"(_fp4_bf16x8_31[1]), "=r"(_fp4_bf16x8_31[2]), "=r"(_fp4_bf16x8_31[3]) : "r"((uint32_t)(braw[7])));
    #else
    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
        "cvt.rn.f16x2.e2m1x2 %0, b0;"
        "cvt.rn.f16x2.e2m1x2 %1, b1;"
        "cvt.rn.f16x2.e2m1x2 %2, b2;"
        "cvt.rn.f16x2.e2m1x2 %3, b3;"
        "}"
        : "=r"(_fp4_bf16x8_31[0]), "=r"(_fp4_bf16x8_31[1]), "=r"(_fp4_bf16x8_31[2]), "=r"(_fp4_bf16x8_31[3]) : "r"((uint32_t)(braw[7])));
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
    a_frag[0] = xa1[0];
    a_frag[1] = ya1[0];
    a_frag[2] = xa1[1];
    a_frag[3] = ya1[1];
    uint32_t _bf16x2_mul_112;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_112) : "r"(_fp4_bf16x8_28[0]), "r"(car0_9));
    b_frag[0] = _bf16x2_mul_112;
    uint32_t _bf16x2_mul_113;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_113) : "r"(_fp4_bf16x8_28[1]), "r"(car0_9));
    b_frag[1] = _bf16x2_mul_113;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_114;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_114) : "r"(_fp4_bf16x8_29[0]), "r"(car1_10));
    b_frag[0] = _bf16x2_mul_114;
    uint32_t _bf16x2_mul_115;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_115) : "r"(_fp4_bf16x8_29[1]), "r"(car1_10));
    b_frag[1] = _bf16x2_mul_115;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_116;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_116) : "r"(_fp4_bf16x8_30[0]), "r"(car2_11));
    b_frag[0] = _bf16x2_mul_116;
    uint32_t _bf16x2_mul_117;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_117) : "r"(_fp4_bf16x8_30[1]), "r"(car2_11));
    b_frag[1] = _bf16x2_mul_117;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_118;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_118) : "r"(_fp4_bf16x8_31[0]), "r"(car3_12));
    b_frag[0] = _bf16x2_mul_118;
    uint32_t _bf16x2_mul_119;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_119) : "r"(_fp4_bf16x8_31[1]), "r"(car3_12));
    b_frag[1] = _bf16x2_mul_119;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    a_frag[0] = xa1[2];
    a_frag[1] = ya1[2];
    a_frag[2] = xa1[3];
    a_frag[3] = ya1[3];
    uint32_t _bf16x2_mul_120;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_120) : "r"(_fp4_bf16x8_28[2]), "r"(car0_9));
    b_frag[0] = _bf16x2_mul_120;
    uint32_t _bf16x2_mul_121;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_121) : "r"(_fp4_bf16x8_28[3]), "r"(car0_9));
    b_frag[1] = _bf16x2_mul_121;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_122;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_122) : "r"(_fp4_bf16x8_29[2]), "r"(car1_10));
    b_frag[0] = _bf16x2_mul_122;
    uint32_t _bf16x2_mul_123;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_123) : "r"(_fp4_bf16x8_29[3]), "r"(car1_10));
    b_frag[1] = _bf16x2_mul_123;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_124;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_124) : "r"(_fp4_bf16x8_30[2]), "r"(car2_11));
    b_frag[0] = _bf16x2_mul_124;
    uint32_t _bf16x2_mul_125;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_125) : "r"(_fp4_bf16x8_30[3]), "r"(car2_11));
    b_frag[1] = _bf16x2_mul_125;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_126;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_126) : "r"(_fp4_bf16x8_31[2]), "r"(car3_12));
    b_frag[0] = _bf16x2_mul_126;
    uint32_t _bf16x2_mul_127;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_127) : "r"(_fp4_bf16x8_31[3]), "r"(car3_12));
    b_frag[1] = _bf16x2_mul_127;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    float part[16];
    int rec_base = ((pair - 1) * 2 + nw) * 512 + lane;
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
        int src_base = nw * 512 + lane;
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
        int src_base_0 = (2 + nw) * 512 + lane;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(smk_part_addr + (unsigned int)(src_base_0 * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 32) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 64) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 96) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 128) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 160) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 192) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 224) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[8])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 256) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[9])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 288) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[10])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 320) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[11])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 352) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[12])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 384) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[13])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 416) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[14])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 448) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[15])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 480) * 4)));
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
        int src_base_1 = (4 + nw) * 512 + lane;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[0])) : "r"(smk_part_addr + (unsigned int)(src_base_1 * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[1])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 32) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[2])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 64) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[3])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 96) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[4])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 128) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[5])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 160) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[6])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 192) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[7])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 224) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[8])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 256) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[9])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 288) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[10])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 320) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[11])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 352) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[12])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 384) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[13])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 416) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[14])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 448) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[15])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 480) * 4)));
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
        int rank = cta_rank;
        int rec_lane = rank * 1024 + lane * 16;
        if (nw * 2 == rank) {
            smk_mail[rec_lane / 4] = acc[0];
            smk_mail[rec_lane / 4 + 1] = acc[1];
            smk_mail[rec_lane / 4 + 2] = acc[2];
            smk_mail[rec_lane / 4 + 3] = acc[3];
            smk_mail[rec_lane / 4 + 128] = acc[4];
            smk_mail[rec_lane / 4 + 129] = acc[5];
            smk_mail[rec_lane / 4 + 130] = acc[6];
            smk_mail[rec_lane / 4 + 131] = acc[7];
        } else {
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(smk_mail_addr + (unsigned int)rec_lane), "r"(nw * 2));
            uint32_t _mbar_peer_16;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_16) : "r"(mail_addr), "r"(nw * 2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])), "r"(_mbar_peer_16) : "memory");
            uint32_t _mbar_peer_17;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_17) : "r"(mail_addr), "r"(nw * 2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 512), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])), "r"(_mbar_peer_17) : "memory");
        }
        if (nw * 2 + 1 == rank) {
            smk_mail[rec_lane / 4] = acc[8];
            smk_mail[rec_lane / 4 + 1] = acc[9];
            smk_mail[rec_lane / 4 + 2] = acc[10];
            smk_mail[rec_lane / 4 + 3] = acc[11];
            smk_mail[rec_lane / 4 + 128] = acc[12];
            smk_mail[rec_lane / 4 + 129] = acc[13];
            smk_mail[rec_lane / 4 + 130] = acc[14];
            smk_mail[rec_lane / 4 + 131] = acc[15];
        } else {
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(smk_mail_addr + (unsigned int)rec_lane), "r"(nw * 2 + 1));
            uint32_t _mbar_peer_18;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_18) : "r"(mail_addr), "r"(nw * 2 + 1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])), "r"(_mbar_peer_18) : "memory");
            uint32_t _mbar_peer_19;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_19) : "r"(mail_addr), "r"(nw * 2 + 1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 512), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])), "r"(_mbar_peer_19) : "memory");
        }
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive(mail_addr);
            }
        }
        if (nw * 2 == rank) {
            mbarrier_wait_cluster_hint(mail_addr, 0, 10000000);
            float rs_vals[32];
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
            float alpha_value = 1.0f;
            int row0 = lane_g;
            int row1 = lane_g + 8;
            int col = off_n + nw * 8 + lane_r * 2;
            if (row0 < M) {
                long long base = (long long)row0 * (long long)N + (long long)col;
                if (col < N) {
                    if (N % 2 < 1) {
                        float _scaled_0[2];
                        const float2 _scale2_20 = {alpha_value, alpha_value};
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(rs_sum)[_ls], _scale2_20);
                        unsigned int _packed_0[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 0], _scaled_0[_lp*2+1 + 0]));
                            _packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(C + (base))[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base)) = (__nv_bfloat16)(rs_sum[0] * alpha_value);
                        if (col + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base + 1)) = (__nv_bfloat16)(rs_sum[1] * alpha_value);
                        }
                    }
                }
                if (col + 16 < N) {
                    if (N % 2 < 1) {
                        float _scaled_1[2];
                        const float2 _scale2_21 = {alpha_value, alpha_value};
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum + 4))[_ls], _scale2_21);
                        unsigned int _packed_1[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 0], _scaled_1[_lp*2+1 + 0]));
                            _packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(C + (base + 16))[0] = reinterpret_cast<unsigned*>(_packed_1)[0];
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base + 16)) = (__nv_bfloat16)(rs_sum[4] * alpha_value);
                        if (col + 16 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base + 16 + 1)) = (__nv_bfloat16)(rs_sum[5] * alpha_value);
                        }
                    }
                }
            }
            if (row1 < M) {
                long long base_1 = (long long)row1 * (long long)N + (long long)col;
                if (col < N) {
                    if (N % 2 < 1) {
                        float _scaled_2[2];
                        const float2 _scale2_22 = {alpha_value, alpha_value};
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            reinterpret_cast<float2*>(_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum + 2))[_ls], _scale2_22);
                        unsigned int _packed_2[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 0], _scaled_2[_lp*2+1 + 0]));
                            _packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(C + (base_1))[0] = reinterpret_cast<unsigned*>(_packed_2)[0];
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base_1)) = (__nv_bfloat16)(rs_sum[2] * alpha_value);
                        if (col + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base_1 + 1)) = (__nv_bfloat16)(rs_sum[3] * alpha_value);
                        }
                    }
                }
                if (col + 16 < N) {
                    if (N % 2 < 1) {
                        float _scaled_3[2];
                        const float2 _scale2_23 = {alpha_value, alpha_value};
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            reinterpret_cast<float2*>(_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum + 6))[_ls], _scale2_23);
                        unsigned int _packed_3[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 0], _scaled_3[_lp*2+1 + 0]));
                            _packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(C + (base_1 + 16))[0] = reinterpret_cast<unsigned*>(_packed_3)[0];
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base_1 + 16)) = (__nv_bfloat16)(rs_sum[6] * alpha_value);
                        if (col + 16 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base_1 + 16 + 1)) = (__nv_bfloat16)(rs_sum[7] * alpha_value);
                        }
                    }
                }
            }
        }
        if (nw * 2 + 1 == rank) {
            mbarrier_wait_cluster_hint(mail_addr, 0, 10000000);
            float rs_vals_1[32];
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
            float alpha_value_1 = 1.0f;
            int row0_1 = lane_g;
            int row1_1 = lane_g + 8;
            int col_1 = off_n + nw * 8 + lane_r * 2;
            if (row0_1 < M) {
                long long base_2 = (long long)row0_1 * (long long)N + (long long)col_1;
                if (col_1 + 32 < N) {
                    if (N % 2 < 1) {
                        float _scaled_4[2];
                        const float2 _scale2_24 = {alpha_value_1, alpha_value_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            reinterpret_cast<float2*>(_scaled_4)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(rs_sum_1)[_ls], _scale2_24);
                        unsigned int _packed_4[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4[_lp*2 + 0], _scaled_4[_lp*2+1 + 0]));
                            _packed_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(C + (base_2 + 32))[0] = reinterpret_cast<unsigned*>(_packed_4)[0];
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base_2 + 32)) = (__nv_bfloat16)(rs_sum_1[0] * alpha_value_1);
                        if (col_1 + 32 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base_2 + 32 + 1)) = (__nv_bfloat16)(rs_sum_1[1] * alpha_value_1);
                        }
                    }
                }
                if (col_1 + 48 < N) {
                    if (N % 2 < 1) {
                        float _scaled_5[2];
                        const float2 _scale2_25 = {alpha_value_1, alpha_value_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            reinterpret_cast<float2*>(_scaled_5)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum_1 + 4))[_ls], _scale2_25);
                        unsigned int _packed_5[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 0], _scaled_5[_lp*2+1 + 0]));
                            _packed_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(C + (base_2 + 48))[0] = reinterpret_cast<unsigned*>(_packed_5)[0];
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base_2 + 48)) = (__nv_bfloat16)(rs_sum_1[4] * alpha_value_1);
                        if (col_1 + 48 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base_2 + 48 + 1)) = (__nv_bfloat16)(rs_sum_1[5] * alpha_value_1);
                        }
                    }
                }
            }
            if (row1_1 < M) {
                long long base_3 = (long long)row1_1 * (long long)N + (long long)col_1;
                if (col_1 + 32 < N) {
                    if (N % 2 < 1) {
                        float _scaled_6[2];
                        const float2 _scale2_26 = {alpha_value_1, alpha_value_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            reinterpret_cast<float2*>(_scaled_6)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum_1 + 2))[_ls], _scale2_26);
                        unsigned int _packed_6[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 0], _scaled_6[_lp*2+1 + 0]));
                            _packed_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(C + (base_3 + 32))[0] = reinterpret_cast<unsigned*>(_packed_6)[0];
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base_3 + 32)) = (__nv_bfloat16)(rs_sum_1[2] * alpha_value_1);
                        if (col_1 + 32 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base_3 + 32 + 1)) = (__nv_bfloat16)(rs_sum_1[3] * alpha_value_1);
                        }
                    }
                }
                if (col_1 + 48 < N) {
                    if (N % 2 < 1) {
                        float _scaled_7[2];
                        const float2 _scale2_27 = {alpha_value_1, alpha_value_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            reinterpret_cast<float2*>(_scaled_7)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((rs_sum_1 + 6))[_ls], _scale2_27);
                        unsigned int _packed_7[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 0], _scaled_7[_lp*2+1 + 0]));
                            _packed_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(C + (base_3 + 48))[0] = reinterpret_cast<unsigned*>(_packed_7)[0];
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base_3 + 48)) = (__nv_bfloat16)(rs_sum_1[6] * alpha_value_1);
                        if (col_1 + 48 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base_3 + 48 + 1)) = (__nv_bfloat16)(rs_sum_1[7] * alpha_value_1);
                        }
                    }
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
