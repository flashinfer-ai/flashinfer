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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_blackwell_bf16_fp4_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMK_A_OFF 1024
#define SMEM_SMK_A_STAGE_BYTES 8192
#define SMEM_SMK_A_STRIDE 17408
#define SMEM_SMK_B0_OFF 9216
#define SMEM_SMK_B0_STAGE_BYTES 2048
#define SMEM_SMK_B0_STRIDE 17408
#define SMEM_SMK_B1_OFF 11264
#define SMEM_SMK_B1_STAGE_BYTES 2048
#define SMEM_SMK_B1_STRIDE 17408
#define SMEM_SMK_B2_OFF 13312
#define SMEM_SMK_B2_STAGE_BYTES 2048
#define SMEM_SMK_B2_STRIDE 17408
#define SMEM_SMK_B3_OFF 15360
#define SMEM_SMK_B3_STAGE_BYTES 2048
#define SMEM_SMK_B3_STRIDE 17408
#define SMEM_SMK_SF_OFF 17408
#define SMEM_SMK_SF_STAGE_BYTES 1024
#define SMEM_SMK_SF_STRIDE 17408
#define SMEM_SMK_PART_OFF 70656
#define SMEM_SMK_PART_STAGE_BYTES 12288
#define SMEM_SMK_PART_STRIDE 12288
#define SMEM_SMK_MAIL_OFF 82944
#define SMEM_SMK_MAIL_STAGE_BYTES 15360
#define SMEM_SMK_MAIL_STRIDE 15360
#define SMEM_TOTAL 98304
#define THREADS 256
#define HAS_ALPHA 1
#define ENABLE_PDL 0

extern "C" {

__global__ __launch_bounds__(THREADS) __cluster_dims__(1,1,4) void
kernel_cake_blackwell_bf16_fp4_7f3a3292f5a452dfe666(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
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
    __nv_bfloat16* smk_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMK_A_OFF);
    const int smk_a_addr = smem + SMEM_SMK_A_OFF;
    uint8_t* smk_b0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMK_B0_OFF);
    const int smk_b0_addr = smem + SMEM_SMK_B0_OFF;
    uint8_t* smk_b1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMK_B1_OFF);
    const int smk_b1_addr = smem + SMEM_SMK_B1_OFF;
    uint8_t* smk_b2 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMK_B2_OFF);
    const int smk_b2_addr = smem + SMEM_SMK_B2_OFF;
    uint8_t* smk_b3 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMK_B3_OFF);
    const int smk_b3_addr = smem + SMEM_SMK_B3_OFF;
    uint8_t* smk_sf = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMK_SF_OFF);
    const int smk_sf_addr = smem + SMEM_SMK_SF_OFF;
    float* smk_part = reinterpret_cast<float*>(smem_raw + SMEM_SMK_PART_OFF);
    const int smk_part_addr = smem + SMEM_SMK_PART_OFF;
    float* smk_mail = reinterpret_cast<float*>(smem_raw + SMEM_SMK_MAIL_OFF);
    const int smk_mail_addr = smem + SMEM_SMK_MAIL_OFF;

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
            mbarrier_expect_tx(smem + 32, 12288);
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // === Task calls (dependency order) ===
    int off_n = blockIdx.x * 64;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(full_addr, 17408);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b0_addr), "l"((&B)), "r"(blockIdx.z * 512), "r"(off_n),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b1_addr), "l"((&B)), "r"(blockIdx.z * 512 + 32), "r"(off_n),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b2_addr), "l"((&B)), "r"(blockIdx.z * 512 + 64), "r"(off_n),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b3_addr), "l"((&B)), "r"(blockIdx.z * 512 + 96), "r"(off_n),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr), "l"((&B_descale)), "r"(blockIdx.z * 64), "r"(off_n),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr + 8, 17408);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b0_addr + 17408), "l"((&B)), "r"(blockIdx.z * 512 + 128), "r"(off_n),
                   "r"(full_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b1_addr + 17408), "l"((&B)), "r"(blockIdx.z * 512 + 160), "r"(off_n),
                   "r"(full_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b2_addr + 17408), "l"((&B)), "r"(blockIdx.z * 512 + 192), "r"(off_n),
                   "r"(full_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b3_addr + 17408), "l"((&B)), "r"(blockIdx.z * 512 + 224), "r"(off_n),
                   "r"(full_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr + 17408), "l"((&B_descale)), "r"(blockIdx.z * 64 + 16), "r"(off_n),
                   "r"(full_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr + 16, 17408);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b0_addr + 34816), "l"((&B)), "r"(blockIdx.z * 512 + 256), "r"(off_n),
                   "r"(full_addr + 16), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b1_addr + 34816), "l"((&B)), "r"(blockIdx.z * 512 + 288), "r"(off_n),
                   "r"(full_addr + 16), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b2_addr + 34816), "l"((&B)), "r"(blockIdx.z * 512 + 320), "r"(off_n),
                   "r"(full_addr + 16), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b3_addr + 34816), "l"((&B)), "r"(blockIdx.z * 512 + 352), "r"(off_n),
                   "r"(full_addr + 16), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr + 34816), "l"((&B_descale)), "r"(blockIdx.z * 64 + 32), "r"(off_n),
                   "r"(full_addr + 16), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr + 24, 17408);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b0_addr + 52224), "l"((&B)), "r"(blockIdx.z * 512 + 384), "r"(off_n),
                   "r"(full_addr + 24), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b1_addr + 52224), "l"((&B)), "r"(blockIdx.z * 512 + 416), "r"(off_n),
                   "r"(full_addr + 24), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b2_addr + 52224), "l"((&B)), "r"(blockIdx.z * 512 + 448), "r"(off_n),
                   "r"(full_addr + 24), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b3_addr + 52224), "l"((&B)), "r"(blockIdx.z * 512 + 480), "r"(off_n),
                   "r"(full_addr + 24), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr + 52224), "l"((&B_descale)), "r"(blockIdx.z * 64 + 48), "r"(off_n),
                   "r"(full_addr + 24), "l"(0x12F0000000000000ULL) : "memory");
        }
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
                :: "r"(smk_a_addr + 17408), "l"((&A)), "r"(0), "r"(0), "r"(blockIdx.z * 16 + 4),
                   "r"(full_addr + 8), "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(smk_a_addr + 34816), "l"((&A)), "r"(0), "r"(0), "r"(blockIdx.z * 16 + 8),
                   "r"(full_addr + 16), "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(smk_a_addr + 52224), "l"((&A)), "r"(0), "r"(0), "r"(blockIdx.z * 16 + 12),
                   "r"(full_addr + 24), "l"(0x14F0000000000000ULL) : "memory");
        }
    }
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
    unsigned int sf_sel = (unsigned int)(lane_r * 17 + 17408);
    unsigned int _phase_full_0 = 0;
    mbarrier_wait(full_addr, _phase_full_0);
    _phase_full_0 ^= 1;
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
        : "r"(smk_b0_addr + (unsigned int)b_lane));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])) : "r"(smk_sf_addr + (unsigned int)sf_lane));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[2])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(2) + 1]))
        : "r"(smk_b0_addr + (unsigned int)b_lane + 512));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[1])) : "r"(smk_sf_addr + (unsigned int)sf_lane + 256));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[4])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(4) + 1]))
        : "r"(smk_b0_addr + (unsigned int)b_lane + 1024));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[2])) : "r"(smk_sf_addr + (unsigned int)sf_lane + 512));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[6])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(6) + 1]))
        : "r"(smk_b0_addr + (unsigned int)b_lane + 1536));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[3])) : "r"(smk_sf_addr + (unsigned int)sf_lane + 768));
    uint32_t _prmt_b32_0;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_0) : "r"(sfw[0]), "r"(sfw[0]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_0 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_0);
    unsigned int car0 = _bf16x2_from_e4m3x2_0;
    uint32_t _prmt_b32_1;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_1) : "r"(sfw[1]), "r"(sfw[1]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_1 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_1);
    unsigned int car1 = _bf16x2_from_e4m3x2_1;
    uint32_t _prmt_b32_2;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_2) : "r"(sfw[2]), "r"(sfw[2]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_2 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_2);
    unsigned int car2 = _bf16x2_from_e4m3x2_2;
    uint32_t _prmt_b32_3;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_3) : "r"(sfw[3]), "r"(sfw[3]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_3 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_3);
    unsigned int car3 = _bf16x2_from_e4m3x2_3;
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
    unsigned int _phase_full_1 = 0;
    mbarrier_wait(full_addr + 8, _phase_full_1);
    _phase_full_1 ^= 1;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa0[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 3]))
        : "r"(smk_a_addr + 17408 + (unsigned int)a_lane0));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 3]))
        : "r"(smk_a_addr + 17408 + (unsigned int)a_lane0 + 1024));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa1[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 3]))
        : "r"(smk_a_addr + 17408 + (unsigned int)a_lane1));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 3]))
        : "r"(smk_a_addr + 17408 + (unsigned int)a_lane1 + 1024));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[0])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(0) + 1]))
        : "r"(smk_b0_addr + 17408 + (unsigned int)b_lane));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])) : "r"(smk_sf_addr + 17408 + (unsigned int)sf_lane));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[2])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(2) + 1]))
        : "r"(smk_b0_addr + 17408 + (unsigned int)b_lane + 512));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[1])) : "r"(smk_sf_addr + 17408 + (unsigned int)sf_lane + 256));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[4])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(4) + 1]))
        : "r"(smk_b0_addr + 17408 + (unsigned int)b_lane + 1024));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[2])) : "r"(smk_sf_addr + 17408 + (unsigned int)sf_lane + 512));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[6])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(6) + 1]))
        : "r"(smk_b0_addr + 17408 + (unsigned int)b_lane + 1536));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[3])) : "r"(smk_sf_addr + 17408 + (unsigned int)sf_lane + 768));
    uint32_t _prmt_b32_4;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_4) : "r"(sfw[0]), "r"(sfw[0]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_4 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_4);
    unsigned int car0_1 = _bf16x2_from_e4m3x2_4;
    uint32_t _prmt_b32_5;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_5) : "r"(sfw[1]), "r"(sfw[1]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_5 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_5);
    unsigned int car1_2 = _bf16x2_from_e4m3x2_5;
    uint32_t _prmt_b32_6;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_6) : "r"(sfw[2]), "r"(sfw[2]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_6 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_6);
    unsigned int car2_3 = _bf16x2_from_e4m3x2_6;
    uint32_t _prmt_b32_7;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_7) : "r"(sfw[3]), "r"(sfw[3]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_7 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_7);
    unsigned int car3_4 = _bf16x2_from_e4m3x2_7;
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
    unsigned int _phase_full_2 = 0;
    mbarrier_wait(full_addr + 16, _phase_full_2);
    _phase_full_2 ^= 1;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa0[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 3]))
        : "r"(smk_a_addr + 34816 + (unsigned int)a_lane0));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 3]))
        : "r"(smk_a_addr + 34816 + (unsigned int)a_lane0 + 1024));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa1[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 3]))
        : "r"(smk_a_addr + 34816 + (unsigned int)a_lane1));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 3]))
        : "r"(smk_a_addr + 34816 + (unsigned int)a_lane1 + 1024));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[0])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(0) + 1]))
        : "r"(smk_b0_addr + 34816 + (unsigned int)b_lane));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])) : "r"(smk_sf_addr + 34816 + (unsigned int)sf_lane));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[2])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(2) + 1]))
        : "r"(smk_b0_addr + 34816 + (unsigned int)b_lane + 512));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[1])) : "r"(smk_sf_addr + 34816 + (unsigned int)sf_lane + 256));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[4])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(4) + 1]))
        : "r"(smk_b0_addr + 34816 + (unsigned int)b_lane + 1024));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[2])) : "r"(smk_sf_addr + 34816 + (unsigned int)sf_lane + 512));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[6])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(6) + 1]))
        : "r"(smk_b0_addr + 34816 + (unsigned int)b_lane + 1536));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[3])) : "r"(smk_sf_addr + 34816 + (unsigned int)sf_lane + 768));
    uint32_t _prmt_b32_8;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_8) : "r"(sfw[0]), "r"(sfw[0]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_8 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_8);
    unsigned int car0_5 = _bf16x2_from_e4m3x2_8;
    uint32_t _prmt_b32_9;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_9) : "r"(sfw[1]), "r"(sfw[1]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_9 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_9);
    unsigned int car1_6 = _bf16x2_from_e4m3x2_9;
    uint32_t _prmt_b32_10;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_10) : "r"(sfw[2]), "r"(sfw[2]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_10 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_10);
    unsigned int car2_7 = _bf16x2_from_e4m3x2_10;
    uint32_t _prmt_b32_11;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_11) : "r"(sfw[3]), "r"(sfw[3]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_11 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_11);
    unsigned int car3_8 = _bf16x2_from_e4m3x2_11;
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
    unsigned int _phase_full_3 = 0;
    mbarrier_wait(full_addr + 24, _phase_full_3);
    _phase_full_3 ^= 1;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa0[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa0[(0) + 3]))
        : "r"(smk_a_addr + 52224 + (unsigned int)a_lane0));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya0[(0) + 3]))
        : "r"(smk_a_addr + 52224 + (unsigned int)a_lane0 + 1024));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&xa1[0])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xa1[(0) + 3]))
        : "r"(smk_a_addr + 52224 + (unsigned int)a_lane1));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&ya1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ya1[(0) + 3]))
        : "r"(smk_a_addr + 52224 + (unsigned int)a_lane1 + 1024));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[0])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(0) + 1]))
        : "r"(smk_b0_addr + 52224 + (unsigned int)b_lane));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[0])) : "r"(smk_sf_addr + 52224 + (unsigned int)sf_lane));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[2])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(2) + 1]))
        : "r"(smk_b0_addr + 52224 + (unsigned int)b_lane + 512));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[1])) : "r"(smk_sf_addr + 52224 + (unsigned int)sf_lane + 256));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[4])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(4) + 1]))
        : "r"(smk_b0_addr + 52224 + (unsigned int)b_lane + 1024));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[2])) : "r"(smk_sf_addr + 52224 + (unsigned int)sf_lane + 512));
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&braw[6])), "=r"(*reinterpret_cast<uint32_t*>(&braw[(6) + 1]))
        : "r"(smk_b0_addr + 52224 + (unsigned int)b_lane + 1536));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfw[3])) : "r"(smk_sf_addr + 52224 + (unsigned int)sf_lane + 768));
    uint32_t _prmt_b32_12;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_12) : "r"(sfw[0]), "r"(sfw[0]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_12 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_12);
    unsigned int car0_9 = _bf16x2_from_e4m3x2_12;
    uint32_t _prmt_b32_13;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_13) : "r"(sfw[1]), "r"(sfw[1]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_13 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_13);
    unsigned int car1_10 = _bf16x2_from_e4m3x2_13;
    uint32_t _prmt_b32_14;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_14) : "r"(sfw[2]), "r"(sfw[2]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_14 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_14);
    unsigned int car2_11 = _bf16x2_from_e4m3x2_14;
    uint32_t _prmt_b32_15;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_15) : "r"(sfw[3]), "r"(sfw[3]), "r"(sf_sel));
    uint32_t _bf16x2_from_e4m3x2_15 = cvt_bf16x2_e4m3x2((uint16_t)_prmt_b32_15);
    unsigned int car3_12 = _bf16x2_from_e4m3x2_15;
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
        if (rank != 0) {
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(mail_addr), "r"(0));
            int slot = ((rank - 1) * 64 + nw * 32 + lane) * 80;
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(smk_mail_addr + (unsigned int)slot), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])), "r"(_mapa_0) : "memory");
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(smk_mail_addr + (unsigned int)slot + 16), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])), "r"(_mapa_0) : "memory");
            uint32_t _mapa_3;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_3) : "r"(smk_mail_addr + (unsigned int)slot + 32), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_3), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])), "r"(_mapa_0) : "memory");
            uint32_t _mapa_4;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_4) : "r"(smk_mail_addr + (unsigned int)slot + 48), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_4), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])), "r"(_mapa_0) : "memory");
        } else {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(mail_addr);
                }
            }
            mbarrier_wait_cluster_hint(mail_addr, 0, 10000000);
            float peer_vals[4];
            int base = (nw * 32 + lane) * 80;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base));
            acc[0] = acc[0] + peer_vals[0];
            acc[1] = acc[1] + peer_vals[1];
            acc[2] = acc[2] + peer_vals[2];
            acc[3] = acc[3] + peer_vals[3];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base + 16));
            acc[4] = acc[4] + peer_vals[0];
            acc[5] = acc[5] + peer_vals[1];
            acc[6] = acc[6] + peer_vals[2];
            acc[7] = acc[7] + peer_vals[3];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base + 32));
            acc[8] = acc[8] + peer_vals[0];
            acc[9] = acc[9] + peer_vals[1];
            acc[10] = acc[10] + peer_vals[2];
            acc[11] = acc[11] + peer_vals[3];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base + 48));
            acc[12] = acc[12] + peer_vals[0];
            acc[13] = acc[13] + peer_vals[1];
            acc[14] = acc[14] + peer_vals[2];
            acc[15] = acc[15] + peer_vals[3];
            int base_0 = (64 + nw * 32 + lane) * 80;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base_0));
            acc[0] = acc[0] + peer_vals[0];
            acc[1] = acc[1] + peer_vals[1];
            acc[2] = acc[2] + peer_vals[2];
            acc[3] = acc[3] + peer_vals[3];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base_0 + 16));
            acc[4] = acc[4] + peer_vals[0];
            acc[5] = acc[5] + peer_vals[1];
            acc[6] = acc[6] + peer_vals[2];
            acc[7] = acc[7] + peer_vals[3];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base_0 + 32));
            acc[8] = acc[8] + peer_vals[0];
            acc[9] = acc[9] + peer_vals[1];
            acc[10] = acc[10] + peer_vals[2];
            acc[11] = acc[11] + peer_vals[3];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base_0 + 48));
            acc[12] = acc[12] + peer_vals[0];
            acc[13] = acc[13] + peer_vals[1];
            acc[14] = acc[14] + peer_vals[2];
            acc[15] = acc[15] + peer_vals[3];
            int base_1 = (128 + nw * 32 + lane) * 80;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base_1));
            acc[0] = acc[0] + peer_vals[0];
            acc[1] = acc[1] + peer_vals[1];
            acc[2] = acc[2] + peer_vals[2];
            acc[3] = acc[3] + peer_vals[3];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base_1 + 16));
            acc[4] = acc[4] + peer_vals[0];
            acc[5] = acc[5] + peer_vals[1];
            acc[6] = acc[6] + peer_vals[2];
            acc[7] = acc[7] + peer_vals[3];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base_1 + 32));
            acc[8] = acc[8] + peer_vals[0];
            acc[9] = acc[9] + peer_vals[1];
            acc[10] = acc[10] + peer_vals[2];
            acc[11] = acc[11] + peer_vals[3];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&peer_vals[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)base_1 + 48));
            acc[12] = acc[12] + peer_vals[0];
            acc[13] = acc[13] + peer_vals[1];
            acc[14] = acc[14] + peer_vals[2];
            acc[15] = acc[15] + peer_vals[3];
            float alpha_value = 1.0f;
            {
                alpha_value = alpha[0];
            }
            int row0 = lane_g;
            int row1 = lane_g + 8;
            int col = off_n + nw * 8 + lane_r * 2;
            if (row0 < M) {
                long long base0 = (long long)row0 * (long long)N + (long long)col;
                if (col < N) {
                    if (N % 2 < 1) {
                        {
                            const float2 _prescale2_0 = {alpha_value, alpha_value};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc[0])[_ps], _prescale2_0);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                acc[0 + _ps] *= alpha_value;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C))[base0]) = _pk;
                        }
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base0)) = __float2bfloat16_rn(acc[0] * alpha_value);
                        if (col + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base0 + 1)) = __float2bfloat16_rn(acc[1] * alpha_value);
                        }
                    }
                }
                if (col + 16 < N) {
                    if (N % 2 < 1) {
                        {
                            const float2 _prescale2_1 = {alpha_value, alpha_value};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc[4])[_ps], _prescale2_1);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                acc[4 + _ps] *= alpha_value;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[4 + 0], acc[4 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C))[base0 + 16]) = _pk;
                        }
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base0 + 16)) = __float2bfloat16_rn(acc[4] * alpha_value);
                        if (col + 16 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base0 + 16 + 1)) = __float2bfloat16_rn(acc[5] * alpha_value);
                        }
                    }
                }
                if (col + 32 < N) {
                    if (N % 2 < 1) {
                        {
                            const float2 _prescale2_2 = {alpha_value, alpha_value};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc[8])[_ps], _prescale2_2);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                acc[8 + _ps] *= alpha_value;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[8 + 0], acc[8 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C))[base0 + 32]) = _pk;
                        }
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base0 + 32)) = __float2bfloat16_rn(acc[8] * alpha_value);
                        if (col + 32 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base0 + 32 + 1)) = __float2bfloat16_rn(acc[9] * alpha_value);
                        }
                    }
                }
                if (col + 48 < N) {
                    if (N % 2 < 1) {
                        {
                            const float2 _prescale2_3 = {alpha_value, alpha_value};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc[12])[_ps], _prescale2_3);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                acc[12 + _ps] *= alpha_value;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[12 + 0], acc[12 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C))[base0 + 48]) = _pk;
                        }
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base0 + 48)) = __float2bfloat16_rn(acc[12] * alpha_value);
                        if (col + 48 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base0 + 48 + 1)) = __float2bfloat16_rn(acc[13] * alpha_value);
                        }
                    }
                }
            }
            if (row1 < M) {
                long long base1 = (long long)row1 * (long long)N + (long long)col;
                if (col < N) {
                    if (N % 2 < 1) {
                        {
                            const float2 _prescale2_4 = {alpha_value, alpha_value};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc[2])[_ps], _prescale2_4);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                acc[2 + _ps] *= alpha_value;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[2 + 0], acc[2 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C))[base1]) = _pk;
                        }
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base1)) = __float2bfloat16_rn(acc[2] * alpha_value);
                        if (col + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base1 + 1)) = __float2bfloat16_rn(acc[3] * alpha_value);
                        }
                    }
                }
                if (col + 16 < N) {
                    if (N % 2 < 1) {
                        {
                            const float2 _prescale2_5 = {alpha_value, alpha_value};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc[6])[_ps], _prescale2_5);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                acc[6 + _ps] *= alpha_value;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[6 + 0], acc[6 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C))[base1 + 16]) = _pk;
                        }
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base1 + 16)) = __float2bfloat16_rn(acc[6] * alpha_value);
                        if (col + 16 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base1 + 16 + 1)) = __float2bfloat16_rn(acc[7] * alpha_value);
                        }
                    }
                }
                if (col + 32 < N) {
                    if (N % 2 < 1) {
                        {
                            const float2 _prescale2_6 = {alpha_value, alpha_value};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc[10])[_ps], _prescale2_6);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                acc[10 + _ps] *= alpha_value;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[10 + 0], acc[10 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C))[base1 + 32]) = _pk;
                        }
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base1 + 32)) = __float2bfloat16_rn(acc[10] * alpha_value);
                        if (col + 32 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base1 + 32 + 1)) = __float2bfloat16_rn(acc[11] * alpha_value);
                        }
                    }
                }
                if (col + 48 < N) {
                    if (N % 2 < 1) {
                        {
                            const float2 _prescale2_7 = {alpha_value, alpha_value};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc[14])[_ps], _prescale2_7);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                acc[14 + _ps] *= alpha_value;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[14 + 0], acc[14 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C))[base1 + 48]) = _pk;
                        }
                    } else {
                        *(reinterpret_cast<__nv_bfloat16*>(C) + (base1 + 48)) = __float2bfloat16_rn(acc[14] * alpha_value);
                        if (col + 48 + 1 < N) {
                            *(reinterpret_cast<__nv_bfloat16*>(C) + (base1 + 48 + 1)) = __float2bfloat16_rn(acc[15] * alpha_value);
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

} // extern "C"
