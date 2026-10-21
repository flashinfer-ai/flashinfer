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
#define SMEM_SMK_FLAG_OFF 78848
#define SMEM_SMK_FLAG_STAGE_BYTES 16
#define SMEM_SMK_FLAG_STRIDE 16
#define SMEM_TOTAL 78976
#define THREADS 256
#define HAS_ALPHA 1
#define ENABLE_PDL 0
#define SCALE_IMAD 0

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_blackwell_bf16_fp4_3402204cb63059a8551f(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, float* __restrict__ partials, unsigned int* __restrict__ completion_u32, int M, int N, int K)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smk_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMK_A_OFF);
    const int smk_a_addr = smem + SMEM_SMK_A_OFF;
    int* smk_b = reinterpret_cast<int*>(smem_raw + SMEM_SMK_B_OFF);
    const int smk_b_addr = smem + SMEM_SMK_B_OFF;
    uint8_t* smk_sf = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMK_SF_OFF);
    const int smk_sf_addr = smem + SMEM_SMK_SF_OFF;
    unsigned int* smk_sfx = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMK_SFX_OFF);
    const int smk_sfx_addr = smem + SMEM_SMK_SFX_OFF;
    float* smk_part = reinterpret_cast<float*>(smem_raw + SMEM_SMK_PART_OFF);
    const int smk_part_addr = smem + SMEM_SMK_PART_OFF;
    int* smk_flag = reinterpret_cast<int*>(smem_raw + SMEM_SMK_FLAG_OFF);
    const int smk_flag_addr = smem + SMEM_SMK_FLAG_OFF;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    int off_n = blockIdx.x * 128;
    int k_blocks = K / 16;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(full_addr, 26624);
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
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr), "l"((&B_descale)), "r"(off_n), "r"(blockIdx.z * 32),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr + 1024), "l"((&B_descale)), "r"(off_n + 64), "r"(blockIdx.z * 32),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr + 8, 26624);
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
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr + 26624), "l"((&B_descale)), "r"(off_n), "r"(blockIdx.z * 32 + 16),
                   "r"(full_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr + 26624 + 1024), "l"((&B_descale)), "r"(off_n + 64), "r"(blockIdx.z * 32 + 16),
                   "r"(full_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
        }
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
    unsigned int _phase_full_0 = 0;
    mbarrier_wait(full_addr, _phase_full_0);
    _phase_full_0 ^= 1;
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
    uint32_t _fp4_dequant_x2_0;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier = _fp4_dequant_x2_0;
    smk_sfx[sfx_dst] = sfx_carrier;
    uint32_t _fp4_dequant_x2_1;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_1 = _fp4_dequant_x2_1;
    smk_sfx[sfx_dst + 1] = sfx_carrier_1;
    uint32_t _fp4_dequant_x2_2;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_2 = _fp4_dequant_x2_2;
    smk_sfx[sfx_dst + 2] = sfx_carrier_2;
    uint32_t _fp4_dequant_x2_3;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_3 = _fp4_dequant_x2_3;
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
    uint32_t _fp4_dequant_x2_4;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_10 = _fp4_dequant_x2_4;
    smk_sfx[sfx_dst_9] = sfx_carrier_10;
    uint32_t _fp4_dequant_x2_5;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_11 = _fp4_dequant_x2_5;
    smk_sfx[sfx_dst_9 + 1] = sfx_carrier_11;
    uint32_t _fp4_dequant_x2_6;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_12 = _fp4_dequant_x2_6;
    smk_sfx[sfx_dst_9 + 2] = sfx_carrier_12;
    uint32_t _fp4_dequant_x2_7;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_13 = _fp4_dequant_x2_7;
    smk_sfx[sfx_dst_9 + 3] = sfx_carrier_13;
    __syncthreads();
    if (k_blocks > blockIdx.z * 32) {
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
    }
    if (k_blocks > blockIdx.z * 32 + 4) {
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
    }
    if (k_blocks > blockIdx.z * 32 + 8) {
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
    }
    if (k_blocks > blockIdx.z * 32 + 12) {
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
    }
    unsigned int _phase_full_1 = 0;
    mbarrier_wait(full_addr + 8, _phase_full_1);
    _phase_full_1 ^= 1;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
    uint32_t _fp4_dequant_x2_8;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_8) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_20 = _fp4_dequant_x2_8;
    smk_sfx[sfx_dst_19] = sfx_carrier_20;
    uint32_t _fp4_dequant_x2_9;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_9) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_21 = _fp4_dequant_x2_9;
    smk_sfx[sfx_dst_19 + 1] = sfx_carrier_21;
    uint32_t _fp4_dequant_x2_10;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_10) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_22 = _fp4_dequant_x2_10;
    smk_sfx[sfx_dst_19 + 2] = sfx_carrier_22;
    uint32_t _fp4_dequant_x2_11;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_11) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_23 = _fp4_dequant_x2_11;
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
    uint32_t _fp4_dequant_x2_12;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_12) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_30 = _fp4_dequant_x2_12;
    smk_sfx[sfx_dst_29] = sfx_carrier_30;
    uint32_t _fp4_dequant_x2_13;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_13) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_31 = _fp4_dequant_x2_13;
    smk_sfx[sfx_dst_29 + 1] = sfx_carrier_31;
    uint32_t _fp4_dequant_x2_14;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_14) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_32 = _fp4_dequant_x2_14;
    smk_sfx[sfx_dst_29 + 2] = sfx_carrier_32;
    uint32_t _fp4_dequant_x2_15;
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
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_15) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
    }
    unsigned int sfx_carrier_33 = _fp4_dequant_x2_15;
    smk_sfx[sfx_dst_29 + 3] = sfx_carrier_33;
    __syncthreads();
    if (k_blocks > blockIdx.z * 32 + 16) {
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
    }
    if (k_blocks > blockIdx.z * 32 + 20) {
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
    }
    if (k_blocks > blockIdx.z * 32 + 24) {
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
    }
    if (k_blocks > blockIdx.z * 32 + 28) {
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
    }
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
        long long plane_elems = (long long)M * (long long)N;
        long long plane = (long long)blockIdx.z * plane_elems;
        int col = pair_off_n + nw * 8 + lane_r * 2;
        int row = lane_g;
        if (row < M) {
            long long base = plane + (long long)row * (long long)N + (long long)col;
            {
                float2 _v2 = make_float2(acc[0 + 0], acc[0 + 1]);
                *reinterpret_cast<float2*>((partials + base) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[4 + 0], acc[4 + 1]);
                *reinterpret_cast<float2*>((partials + (base + 16)) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[8 + 0], acc[8 + 1]);
                *reinterpret_cast<float2*>((partials + (base + 32)) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[12 + 0], acc[12 + 1]);
                *reinterpret_cast<float2*>((partials + (base + 48)) + 0) = _v2;
            }
        }
        int row_0 = lane_g + 8;
        if (row_0 < M) {
            long long base_1 = plane + (long long)row_0 * (long long)N + (long long)col;
            {
                float2 _v2 = make_float2(acc[2 + 0], acc[2 + 1]);
                *reinterpret_cast<float2*>((partials + base_1) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[6 + 0], acc[6 + 1]);
                *reinterpret_cast<float2*>((partials + (base_1 + 16)) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[10 + 0], acc[10 + 1]);
                *reinterpret_cast<float2*>((partials + (base_1 + 32)) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[14 + 0], acc[14 + 1]);
                *reinterpret_cast<float2*>((partials + (base_1 + 48)) + 0) = _v2;
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        int old = 0;
        if (lane == 0) {
            unsigned int _atomic_old_0;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                : "=r"(_atomic_old_0) : "l"(&completion_u32[blockIdx.x]), "r"(static_cast<uint32_t>(1)) : "memory");
            unsigned int old_u = _atomic_old_0;
            old = (int)old_u;
            smk_flag[0] = old;
        }
    }
    __syncthreads();
    int flag[1];
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&flag[0])) : "r"(smk_flag_addr));
    if (flag[0] == 7) {
        float alpha_value = 1.0f;
        {
            alpha_value = alpha[0];
        }
        long long plane_elems_1 = (long long)M * (long long)N;
        int tid_0 = tid;
        {
            int chunk = tid_0;
            int row_1 = chunk >> 5;
            if (row_1 < M) {
                int col_1 = off_n + (chunk & 31) * 4;
                long long base_2 = (long long)row_1 * (long long)N + (long long)col_1;
                float fin[4];
                float _vec_load_0[4];
                {
                    unsigned _v4_0_0;
                    unsigned _v4_0_1;
                    unsigned _v4_0_2;
                    unsigned _v4_0_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3) : "l"((const void*)(partials + (base_2))) : "memory");
                    _vec_load_0[0 + 0] = __uint_as_float(_v4_0_0);
                    _vec_load_0[0 + 1] = __uint_as_float(_v4_0_1);
                    _vec_load_0[0 + 2] = __uint_as_float(_v4_0_2);
                    _vec_load_0[0 + 3] = __uint_as_float(_v4_0_3);
                }
                fin[0] = _vec_load_0[0];
                fin[1] = _vec_load_0[1];
                fin[2] = _vec_load_0[2];
                fin[3] = _vec_load_0[3];
                float _vec_load_1[4];
                {
                    unsigned _v4_1_0;
                    unsigned _v4_1_1;
                    unsigned _v4_1_2;
                    unsigned _v4_1_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3) : "l"((const void*)(partials + (base_2 + plane_elems_1))) : "memory");
                    _vec_load_1[0 + 0] = __uint_as_float(_v4_1_0);
                    _vec_load_1[0 + 1] = __uint_as_float(_v4_1_1);
                    _vec_load_1[0 + 2] = __uint_as_float(_v4_1_2);
                    _vec_load_1[0 + 3] = __uint_as_float(_v4_1_3);
                }
                fin[0] = fin[0] + _vec_load_1[0];
                fin[1] = fin[1] + _vec_load_1[1];
                fin[2] = fin[2] + _vec_load_1[2];
                fin[3] = fin[3] + _vec_load_1[3];
                float _vec_load_2[4];
                {
                    unsigned _v4_2_0;
                    unsigned _v4_2_1;
                    unsigned _v4_2_2;
                    unsigned _v4_2_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_2_0), "=r"(_v4_2_1), "=r"(_v4_2_2), "=r"(_v4_2_3) : "l"((const void*)(partials + (base_2 + 2 * plane_elems_1))) : "memory");
                    _vec_load_2[0 + 0] = __uint_as_float(_v4_2_0);
                    _vec_load_2[0 + 1] = __uint_as_float(_v4_2_1);
                    _vec_load_2[0 + 2] = __uint_as_float(_v4_2_2);
                    _vec_load_2[0 + 3] = __uint_as_float(_v4_2_3);
                }
                fin[0] = fin[0] + _vec_load_2[0];
                fin[1] = fin[1] + _vec_load_2[1];
                fin[2] = fin[2] + _vec_load_2[2];
                fin[3] = fin[3] + _vec_load_2[3];
                float _vec_load_3[4];
                {
                    unsigned _v4_3_0;
                    unsigned _v4_3_1;
                    unsigned _v4_3_2;
                    unsigned _v4_3_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_3_0), "=r"(_v4_3_1), "=r"(_v4_3_2), "=r"(_v4_3_3) : "l"((const void*)(partials + (base_2 + 3 * plane_elems_1))) : "memory");
                    _vec_load_3[0 + 0] = __uint_as_float(_v4_3_0);
                    _vec_load_3[0 + 1] = __uint_as_float(_v4_3_1);
                    _vec_load_3[0 + 2] = __uint_as_float(_v4_3_2);
                    _vec_load_3[0 + 3] = __uint_as_float(_v4_3_3);
                }
                fin[0] = fin[0] + _vec_load_3[0];
                fin[1] = fin[1] + _vec_load_3[1];
                fin[2] = fin[2] + _vec_load_3[2];
                fin[3] = fin[3] + _vec_load_3[3];
                float _vec_load_4[4];
                {
                    unsigned _v4_4_0;
                    unsigned _v4_4_1;
                    unsigned _v4_4_2;
                    unsigned _v4_4_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_4_0), "=r"(_v4_4_1), "=r"(_v4_4_2), "=r"(_v4_4_3) : "l"((const void*)(partials + (base_2 + 4 * plane_elems_1))) : "memory");
                    _vec_load_4[0 + 0] = __uint_as_float(_v4_4_0);
                    _vec_load_4[0 + 1] = __uint_as_float(_v4_4_1);
                    _vec_load_4[0 + 2] = __uint_as_float(_v4_4_2);
                    _vec_load_4[0 + 3] = __uint_as_float(_v4_4_3);
                }
                fin[0] = fin[0] + _vec_load_4[0];
                fin[1] = fin[1] + _vec_load_4[1];
                fin[2] = fin[2] + _vec_load_4[2];
                fin[3] = fin[3] + _vec_load_4[3];
                float _vec_load_5[4];
                {
                    unsigned _v4_5_0;
                    unsigned _v4_5_1;
                    unsigned _v4_5_2;
                    unsigned _v4_5_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_5_0), "=r"(_v4_5_1), "=r"(_v4_5_2), "=r"(_v4_5_3) : "l"((const void*)(partials + (base_2 + 5 * plane_elems_1))) : "memory");
                    _vec_load_5[0 + 0] = __uint_as_float(_v4_5_0);
                    _vec_load_5[0 + 1] = __uint_as_float(_v4_5_1);
                    _vec_load_5[0 + 2] = __uint_as_float(_v4_5_2);
                    _vec_load_5[0 + 3] = __uint_as_float(_v4_5_3);
                }
                fin[0] = fin[0] + _vec_load_5[0];
                fin[1] = fin[1] + _vec_load_5[1];
                fin[2] = fin[2] + _vec_load_5[2];
                fin[3] = fin[3] + _vec_load_5[3];
                float _vec_load_6[4];
                {
                    unsigned _v4_6_0;
                    unsigned _v4_6_1;
                    unsigned _v4_6_2;
                    unsigned _v4_6_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_6_0), "=r"(_v4_6_1), "=r"(_v4_6_2), "=r"(_v4_6_3) : "l"((const void*)(partials + (base_2 + 6 * plane_elems_1))) : "memory");
                    _vec_load_6[0 + 0] = __uint_as_float(_v4_6_0);
                    _vec_load_6[0 + 1] = __uint_as_float(_v4_6_1);
                    _vec_load_6[0 + 2] = __uint_as_float(_v4_6_2);
                    _vec_load_6[0 + 3] = __uint_as_float(_v4_6_3);
                }
                fin[0] = fin[0] + _vec_load_6[0];
                fin[1] = fin[1] + _vec_load_6[1];
                fin[2] = fin[2] + _vec_load_6[2];
                fin[3] = fin[3] + _vec_load_6[3];
                float _vec_load_7[4];
                {
                    unsigned _v4_7_0;
                    unsigned _v4_7_1;
                    unsigned _v4_7_2;
                    unsigned _v4_7_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_7_0), "=r"(_v4_7_1), "=r"(_v4_7_2), "=r"(_v4_7_3) : "l"((const void*)(partials + (base_2 + 7 * plane_elems_1))) : "memory");
                    _vec_load_7[0 + 0] = __uint_as_float(_v4_7_0);
                    _vec_load_7[0 + 1] = __uint_as_float(_v4_7_1);
                    _vec_load_7[0 + 2] = __uint_as_float(_v4_7_2);
                    _vec_load_7[0 + 3] = __uint_as_float(_v4_7_3);
                }
                fin[0] = fin[0] + _vec_load_7[0];
                fin[1] = fin[1] + _vec_load_7[1];
                fin[2] = fin[2] + _vec_load_7[2];
                fin[3] = fin[3] + _vec_load_7[3];
                {
                    const float2 _prescale2_8 = {alpha_value, alpha_value};
                    #if __CUDA_ARCH__ >= 1000
                    #pragma unroll
                    for (int _ps = 0; _ps < 2; _ps++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&fin[0])[_ps], _prescale2_8);
                    #else
                    #pragma unroll
                    for (int _ps = 0; _ps < 4; _ps++)
                        fin[0 + _ps] *= alpha_value;
                    #endif
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(fin[0 + 0], fin[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(fin[0 + 2], fin[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(C + base_2))[0]) = _pk2;
                }
            }
            int chunk_0 = tid_0 + 256;
            int row_1_1 = chunk_0 >> 5;
            if (row_1_1 < M) {
                int col_2 = off_n + (chunk_0 & 31) * 4;
                long long base_3 = (long long)row_1_1 * (long long)N + (long long)col_2;
                float fin_1[4];
                float _vec_load_8[4];
                {
                    unsigned _v4_9_0;
                    unsigned _v4_9_1;
                    unsigned _v4_9_2;
                    unsigned _v4_9_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_9_0), "=r"(_v4_9_1), "=r"(_v4_9_2), "=r"(_v4_9_3) : "l"((const void*)(partials + (base_3))) : "memory");
                    _vec_load_8[0 + 0] = __uint_as_float(_v4_9_0);
                    _vec_load_8[0 + 1] = __uint_as_float(_v4_9_1);
                    _vec_load_8[0 + 2] = __uint_as_float(_v4_9_2);
                    _vec_load_8[0 + 3] = __uint_as_float(_v4_9_3);
                }
                fin_1[0] = _vec_load_8[0];
                fin_1[1] = _vec_load_8[1];
                fin_1[2] = _vec_load_8[2];
                fin_1[3] = _vec_load_8[3];
                float _vec_load_9[4];
                {
                    unsigned _v4_10_0;
                    unsigned _v4_10_1;
                    unsigned _v4_10_2;
                    unsigned _v4_10_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_10_0), "=r"(_v4_10_1), "=r"(_v4_10_2), "=r"(_v4_10_3) : "l"((const void*)(partials + (base_3 + plane_elems_1))) : "memory");
                    _vec_load_9[0 + 0] = __uint_as_float(_v4_10_0);
                    _vec_load_9[0 + 1] = __uint_as_float(_v4_10_1);
                    _vec_load_9[0 + 2] = __uint_as_float(_v4_10_2);
                    _vec_load_9[0 + 3] = __uint_as_float(_v4_10_3);
                }
                fin_1[0] = fin_1[0] + _vec_load_9[0];
                fin_1[1] = fin_1[1] + _vec_load_9[1];
                fin_1[2] = fin_1[2] + _vec_load_9[2];
                fin_1[3] = fin_1[3] + _vec_load_9[3];
                float _vec_load_10[4];
                {
                    unsigned _v4_11_0;
                    unsigned _v4_11_1;
                    unsigned _v4_11_2;
                    unsigned _v4_11_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_11_0), "=r"(_v4_11_1), "=r"(_v4_11_2), "=r"(_v4_11_3) : "l"((const void*)(partials + (base_3 + 2 * plane_elems_1))) : "memory");
                    _vec_load_10[0 + 0] = __uint_as_float(_v4_11_0);
                    _vec_load_10[0 + 1] = __uint_as_float(_v4_11_1);
                    _vec_load_10[0 + 2] = __uint_as_float(_v4_11_2);
                    _vec_load_10[0 + 3] = __uint_as_float(_v4_11_3);
                }
                fin_1[0] = fin_1[0] + _vec_load_10[0];
                fin_1[1] = fin_1[1] + _vec_load_10[1];
                fin_1[2] = fin_1[2] + _vec_load_10[2];
                fin_1[3] = fin_1[3] + _vec_load_10[3];
                float _vec_load_11[4];
                {
                    unsigned _v4_12_0;
                    unsigned _v4_12_1;
                    unsigned _v4_12_2;
                    unsigned _v4_12_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_12_0), "=r"(_v4_12_1), "=r"(_v4_12_2), "=r"(_v4_12_3) : "l"((const void*)(partials + (base_3 + 3 * plane_elems_1))) : "memory");
                    _vec_load_11[0 + 0] = __uint_as_float(_v4_12_0);
                    _vec_load_11[0 + 1] = __uint_as_float(_v4_12_1);
                    _vec_load_11[0 + 2] = __uint_as_float(_v4_12_2);
                    _vec_load_11[0 + 3] = __uint_as_float(_v4_12_3);
                }
                fin_1[0] = fin_1[0] + _vec_load_11[0];
                fin_1[1] = fin_1[1] + _vec_load_11[1];
                fin_1[2] = fin_1[2] + _vec_load_11[2];
                fin_1[3] = fin_1[3] + _vec_load_11[3];
                float _vec_load_12[4];
                {
                    unsigned _v4_13_0;
                    unsigned _v4_13_1;
                    unsigned _v4_13_2;
                    unsigned _v4_13_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_13_0), "=r"(_v4_13_1), "=r"(_v4_13_2), "=r"(_v4_13_3) : "l"((const void*)(partials + (base_3 + 4 * plane_elems_1))) : "memory");
                    _vec_load_12[0 + 0] = __uint_as_float(_v4_13_0);
                    _vec_load_12[0 + 1] = __uint_as_float(_v4_13_1);
                    _vec_load_12[0 + 2] = __uint_as_float(_v4_13_2);
                    _vec_load_12[0 + 3] = __uint_as_float(_v4_13_3);
                }
                fin_1[0] = fin_1[0] + _vec_load_12[0];
                fin_1[1] = fin_1[1] + _vec_load_12[1];
                fin_1[2] = fin_1[2] + _vec_load_12[2];
                fin_1[3] = fin_1[3] + _vec_load_12[3];
                float _vec_load_13[4];
                {
                    unsigned _v4_14_0;
                    unsigned _v4_14_1;
                    unsigned _v4_14_2;
                    unsigned _v4_14_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_14_0), "=r"(_v4_14_1), "=r"(_v4_14_2), "=r"(_v4_14_3) : "l"((const void*)(partials + (base_3 + 5 * plane_elems_1))) : "memory");
                    _vec_load_13[0 + 0] = __uint_as_float(_v4_14_0);
                    _vec_load_13[0 + 1] = __uint_as_float(_v4_14_1);
                    _vec_load_13[0 + 2] = __uint_as_float(_v4_14_2);
                    _vec_load_13[0 + 3] = __uint_as_float(_v4_14_3);
                }
                fin_1[0] = fin_1[0] + _vec_load_13[0];
                fin_1[1] = fin_1[1] + _vec_load_13[1];
                fin_1[2] = fin_1[2] + _vec_load_13[2];
                fin_1[3] = fin_1[3] + _vec_load_13[3];
                float _vec_load_14[4];
                {
                    unsigned _v4_15_0;
                    unsigned _v4_15_1;
                    unsigned _v4_15_2;
                    unsigned _v4_15_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_15_0), "=r"(_v4_15_1), "=r"(_v4_15_2), "=r"(_v4_15_3) : "l"((const void*)(partials + (base_3 + 6 * plane_elems_1))) : "memory");
                    _vec_load_14[0 + 0] = __uint_as_float(_v4_15_0);
                    _vec_load_14[0 + 1] = __uint_as_float(_v4_15_1);
                    _vec_load_14[0 + 2] = __uint_as_float(_v4_15_2);
                    _vec_load_14[0 + 3] = __uint_as_float(_v4_15_3);
                }
                fin_1[0] = fin_1[0] + _vec_load_14[0];
                fin_1[1] = fin_1[1] + _vec_load_14[1];
                fin_1[2] = fin_1[2] + _vec_load_14[2];
                fin_1[3] = fin_1[3] + _vec_load_14[3];
                float _vec_load_15[4];
                {
                    unsigned _v4_16_0;
                    unsigned _v4_16_1;
                    unsigned _v4_16_2;
                    unsigned _v4_16_3;
                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_v4_16_0), "=r"(_v4_16_1), "=r"(_v4_16_2), "=r"(_v4_16_3) : "l"((const void*)(partials + (base_3 + 7 * plane_elems_1))) : "memory");
                    _vec_load_15[0 + 0] = __uint_as_float(_v4_16_0);
                    _vec_load_15[0 + 1] = __uint_as_float(_v4_16_1);
                    _vec_load_15[0 + 2] = __uint_as_float(_v4_16_2);
                    _vec_load_15[0 + 3] = __uint_as_float(_v4_16_3);
                }
                fin_1[0] = fin_1[0] + _vec_load_15[0];
                fin_1[1] = fin_1[1] + _vec_load_15[1];
                fin_1[2] = fin_1[2] + _vec_load_15[2];
                fin_1[3] = fin_1[3] + _vec_load_15[3];
                {
                    const float2 _prescale2_17 = {alpha_value, alpha_value};
                    #if __CUDA_ARCH__ >= 1000
                    #pragma unroll
                    for (int _ps = 0; _ps < 2; _ps++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&fin_1[0])[_ps], _prescale2_17);
                    #else
                    #pragma unroll
                    for (int _ps = 0; _ps < 4; _ps++)
                        fin_1[0 + _ps] *= alpha_value;
                    #endif
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(fin_1[0 + 0], fin_1[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(fin_1[0 + 2], fin_1[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(C + base_3))[0]) = _pk2;
                }
            }
        }
        if (tid == 0) {
            completion_u32[blockIdx.x] = 0;
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
