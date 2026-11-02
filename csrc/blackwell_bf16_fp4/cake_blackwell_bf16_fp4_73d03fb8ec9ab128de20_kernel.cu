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
#define SMEM_SMK_A_STAGE_BYTES 16384
#define SMEM_SMK_A_STRIDE 25600
#define SMEM_SMK_B_OFF 17408
#define SMEM_SMK_B_STAGE_BYTES 8192
#define SMEM_SMK_B_STRIDE 25600
#define SMEM_SMK_SF_OFF 25600
#define SMEM_SMK_SF_STAGE_BYTES 1024
#define SMEM_SMK_SF_STRIDE 25600
#define SMEM_SMK_SFX_OFF 52224
#define SMEM_SMK_SFX_STAGE_BYTES 4096
#define SMEM_SMK_SFX_STRIDE 4096
#define SMEM_SMK_PART_OFF 60416
#define SMEM_SMK_PART_STAGE_BYTES 24576
#define SMEM_SMK_PART_STRIDE 24576
#define SMEM_SMK_FLAG_OFF 84992
#define SMEM_SMK_FLAG_STAGE_BYTES 16
#define SMEM_SMK_FLAG_STRIDE 16
#define SMEM_TOTAL 85120
#define THREADS 256
#define HAS_ALPHA 0
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_blackwell_bf16_fp4_73d03fb8ec9ab128de20(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, float* __restrict__ partials, unsigned int* __restrict__ completion_u32, int M, int N, int K)
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
    #define sf_full_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smk_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smk_a_addr = smem + 1024;
    int* smk_b = reinterpret_cast<int*>(smem_raw + 17408);
    const int smk_b_addr = smem + 17408;
    uint8_t* smk_sf = reinterpret_cast<uint8_t*>(smem_raw + 25600);
    const int smk_sf_addr = smem + 25600;
    unsigned int* smk_sfx = reinterpret_cast<unsigned int*>(smem_raw + 52224);
    const int smk_sfx_addr = smem + 52224;
    float* smk_part = reinterpret_cast<float*>(smem_raw + 60416);
    const int smk_part_addr = smem + 60416;
    int* smk_flag = reinterpret_cast<int*>(smem_raw + 84992);
    const int smk_flag_addr = smem + 84992;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // sf_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    int off_n = blockIdx.x * 64;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(sf_full_addr, 2048);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr), "l"((&B_descale)), "r"(off_n), "r"(blockIdx.z * 32),
                   "r"(sf_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr + 25600), "l"((&B_descale)), "r"(off_n), "r"(blockIdx.z * 32 + 16),
                   "r"(sf_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr, 24576);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b_addr), "l"((&B)), "r"(off_n * 2), "r"(blockIdx.z * 32),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr + 8, 24576);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b_addr + 25600), "l"((&B)), "r"(off_n * 2), "r"(blockIdx.z * 32 + 16),
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
                :: "r"(smk_a_addr + 25600), "l"((&A)), "r"(0), "r"(0), "r"(blockIdx.z * 8 + 4),
                   "r"(full_addr + 8), "l"(0x14F0000000000000ULL) : "memory");
        }
    }
    unsigned int a_frag[4];
    unsigned int a_frag_hi[4];
    unsigned int raw[2];
    unsigned int sf_word[4];
    unsigned int carriers[4];
    unsigned int sfx_word[4];
    unsigned int b_frag[2];
    float acc[32];
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
    acc[16] = 0.0f;
    acc[17] = 0.0f;
    acc[18] = 0.0f;
    acc[19] = 0.0f;
    acc[20] = 0.0f;
    acc[21] = 0.0f;
    acc[22] = 0.0f;
    acc[23] = 0.0f;
    acc[24] = 0.0f;
    acc[25] = 0.0f;
    acc[26] = 0.0f;
    acc[27] = 0.0f;
    acc[28] = 0.0f;
    acc[29] = 0.0f;
    acc[30] = 0.0f;
    acc[31] = 0.0f;
    int warp_0 = warp;
    int nw = warp_0 % 2;
    int pair = warp_0 / 2;
    int lane_g = lane / 4;
    int lane_r = lane % 4;
    int a_row = lane % 16;
    int a_col = lane / 16 * 8;
    int b_lane_off = nw * 256 + lane * 8;
    int sf_lane_off = nw * 8 + (lane_g & 4);
    int sf_shift = (lane_g & 3) * 8;
    int sfx_lane_off = (nw * 8 + lane_g) * 16;
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
    int sfx_rec_4 = tid;
    int sfx_kb_5 = sfx_rec_4 >> 4;
    int sfx_nlo_6 = sfx_rec_4 & 15;
    int sfx_src_7 = smk_sf_addr + 25600 + (unsigned int)(sfx_kb_5 * 64) + (unsigned int)(sfx_nlo_6 & 12);
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[0])) : "r"(sfx_src_7));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[1])) : "r"(sfx_src_7 + 16));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[2])) : "r"(sfx_src_7 + 32));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&sfx_word[3])) : "r"(sfx_src_7 + 48));
    int sfx_shift_8 = (sfx_nlo_6 & 3) * 8;
    int sfx_dst_9 = (sfx_kb_5 * 16 + sfx_nlo_6) * 4 + 1024;
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
    __syncthreads();
    mbarrier_wait(full_addr, 0);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)((pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (pair * 16 + a_col) % 64 * 2 ^ ((pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag_hi[0]), "=r"(a_frag_hi[1]), "=r"(a_frag_hi[2]), "=r"(a_frag_hi[3])
        : "r"((smk_a_addr + (unsigned int)((pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (pair * 16 + a_col) % 64 * 2 ^ ((pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(_fp4_bf16x8_0[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_2;
    uint32_t _bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(_fp4_bf16x8_0[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_3;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(_fp4_bf16x8_1[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_4;
    uint32_t _bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(_fp4_bf16x8_1[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_5;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(_fp4_bf16x8_1[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_6;
    uint32_t _bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(_fp4_bf16x8_1[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_7;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)((64 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (64 + pair * 16 + a_col) % 64 * 2 ^ ((64 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (64 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag_hi[0]), "=r"(a_frag_hi[1]), "=r"(a_frag_hi[2]), "=r"(a_frag_hi[3])
        : "r"((smk_a_addr + (unsigned int)((64 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (64 + pair * 16 + a_col) % 64 * 2 ^ ((64 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (64 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 2048 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 1024 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(_fp4_bf16x8_2[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_10;
    uint32_t _bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(_fp4_bf16x8_2[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_11;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(_fp4_bf16x8_3[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_12;
    uint32_t _bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(_fp4_bf16x8_3[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_13;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(_fp4_bf16x8_3[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_14;
    uint32_t _bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(_fp4_bf16x8_3[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_15;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)((128 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (128 + pair * 16 + a_col) % 64 * 2 ^ ((128 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (128 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag_hi[0]), "=r"(a_frag_hi[1]), "=r"(a_frag_hi[2]), "=r"(a_frag_hi[3])
        : "r"((smk_a_addr + (unsigned int)((128 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (128 + pair * 16 + a_col) % 64 * 2 ^ ((128 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (128 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 4096 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 2048 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(_fp4_bf16x8_4[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_18;
    uint32_t _bf16x2_mul_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(_fp4_bf16x8_4[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_19;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(_fp4_bf16x8_5[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_20;
    uint32_t _bf16x2_mul_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(_fp4_bf16x8_5[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_21;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(_fp4_bf16x8_5[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_22;
    uint32_t _bf16x2_mul_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(_fp4_bf16x8_5[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_23;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + (unsigned int)((192 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (192 + pair * 16 + a_col) % 64 * 2 ^ ((192 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (192 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag_hi[0]), "=r"(a_frag_hi[1]), "=r"(a_frag_hi[2]), "=r"(a_frag_hi[3])
        : "r"((smk_a_addr + (unsigned int)((192 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (192 + pair * 16 + a_col) % 64 * 2 ^ ((192 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (192 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 6144 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 3072 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(_fp4_bf16x8_6[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_26;
    uint32_t _bf16x2_mul_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(_fp4_bf16x8_6[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_27;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(_fp4_bf16x8_7[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_28;
    uint32_t _bf16x2_mul_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(_fp4_bf16x8_7[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_29;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(_fp4_bf16x8_7[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_30;
    uint32_t _bf16x2_mul_31;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(_fp4_bf16x8_7[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_31;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    mbarrier_wait(full_addr + 8, 0);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 25600 + (unsigned int)((pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (pair * 16 + a_col) % 64 * 2 ^ ((pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag_hi[0]), "=r"(a_frag_hi[1]), "=r"(a_frag_hi[2]), "=r"(a_frag_hi[3])
        : "r"((smk_a_addr + 25600 + (unsigned int)((pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (pair * 16 + a_col) % 64 * 2 ^ ((pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 25600 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 4096 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_34;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(_fp4_bf16x8_8[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_34;
    uint32_t _bf16x2_mul_35;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(_fp4_bf16x8_8[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_35;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_36;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(_fp4_bf16x8_9[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_36;
    uint32_t _bf16x2_mul_37;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(_fp4_bf16x8_9[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_37;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_38;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(_fp4_bf16x8_9[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_38;
    uint32_t _bf16x2_mul_39;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(_fp4_bf16x8_9[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_39;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 25600 + (unsigned int)((64 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (64 + pair * 16 + a_col) % 64 * 2 ^ ((64 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (64 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag_hi[0]), "=r"(a_frag_hi[1]), "=r"(a_frag_hi[2]), "=r"(a_frag_hi[3])
        : "r"((smk_a_addr + 25600 + (unsigned int)((64 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (64 + pair * 16 + a_col) % 64 * 2 ^ ((64 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (64 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 25600 + 2048 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 4096 + 1024 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_42;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(_fp4_bf16x8_10[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_42;
    uint32_t _bf16x2_mul_43;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(_fp4_bf16x8_10[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_43;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_44;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(_fp4_bf16x8_11[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_44;
    uint32_t _bf16x2_mul_45;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(_fp4_bf16x8_11[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_45;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_46;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(_fp4_bf16x8_11[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_46;
    uint32_t _bf16x2_mul_47;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(_fp4_bf16x8_11[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_47;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 25600 + (unsigned int)((128 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (128 + pair * 16 + a_col) % 64 * 2 ^ ((128 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (128 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag_hi[0]), "=r"(a_frag_hi[1]), "=r"(a_frag_hi[2]), "=r"(a_frag_hi[3])
        : "r"((smk_a_addr + 25600 + (unsigned int)((128 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (128 + pair * 16 + a_col) % 64 * 2 ^ ((128 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (128 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 25600 + 4096 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 4096 + 2048 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_50;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(_fp4_bf16x8_12[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_50;
    uint32_t _bf16x2_mul_51;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(_fp4_bf16x8_12[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_51;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_52;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(_fp4_bf16x8_13[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_52;
    uint32_t _bf16x2_mul_53;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(_fp4_bf16x8_13[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_53;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_54;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(_fp4_bf16x8_13[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_54;
    uint32_t _bf16x2_mul_55;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(_fp4_bf16x8_13[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_55;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"((smk_a_addr + 25600 + (unsigned int)((192 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (192 + pair * 16 + a_col) % 64 * 2 ^ ((192 + pair * 16 + a_col) / 64 * 4096 + a_row * 128 + (192 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag_hi[0]), "=r"(a_frag_hi[1]), "=r"(a_frag_hi[2]), "=r"(a_frag_hi[3])
        : "r"((smk_a_addr + 25600 + (unsigned int)((192 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (192 + pair * 16 + a_col) % 64 * 2 ^ ((192 + pair * 16 + a_col) / 64 * 4096 + (a_row + 16) * 128 + (192 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
        : "memory");
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
        : "r"(smk_b_addr + 25600 + 6144 + (unsigned int)(pair * 512) + (unsigned int)b_lane_off));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&carriers[0])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carriers[(0) + 3]))
        : "r"(smk_sfx_addr + 4096 + 3072 + (unsigned int)(pair * 256) + (unsigned int)sfx_lane_off));
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
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_58;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(_fp4_bf16x8_14[2]), "r"(carriers[1]));
    b_frag[0] = _bf16x2_mul_58;
    uint32_t _bf16x2_mul_59;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(_fp4_bf16x8_14[3]), "r"(carriers[1]));
    b_frag[1] = _bf16x2_mul_59;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_60;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(_fp4_bf16x8_15[0]), "r"(carriers[2]));
    b_frag[0] = _bf16x2_mul_60;
    uint32_t _bf16x2_mul_61;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(_fp4_bf16x8_15[1]), "r"(carriers[2]));
    b_frag[1] = _bf16x2_mul_61;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    uint32_t _bf16x2_mul_62;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(_fp4_bf16x8_15[2]), "r"(carriers[3]));
    b_frag[0] = _bf16x2_mul_62;
    uint32_t _bf16x2_mul_63;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(_fp4_bf16x8_15[3]), "r"(carriers[3]));
    b_frag[1] = _bf16x2_mul_63;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag_hi[0]), "r"(a_frag_hi[1]), "r"(a_frag_hi[2]), "r"(a_frag_hi[3]), "r"(b_frag[0]), "r"(b_frag[1]));
    float part[32];
    int rec_base = ((pair - 1) * 2 + nw) * 1024 + lane;
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
        smk_part[rec_base + 512] = acc[16];
        smk_part[rec_base + 544] = acc[17];
        smk_part[rec_base + 576] = acc[18];
        smk_part[rec_base + 608] = acc[19];
        smk_part[rec_base + 640] = acc[20];
        smk_part[rec_base + 672] = acc[21];
        smk_part[rec_base + 704] = acc[22];
        smk_part[rec_base + 736] = acc[23];
        smk_part[rec_base + 768] = acc[24];
        smk_part[rec_base + 800] = acc[25];
        smk_part[rec_base + 832] = acc[26];
        smk_part[rec_base + 864] = acc[27];
        smk_part[rec_base + 896] = acc[28];
        smk_part[rec_base + 928] = acc[29];
        smk_part[rec_base + 960] = acc[30];
        smk_part[rec_base + 992] = acc[31];
    }
    __syncthreads();
    if (pair < 1) {
        int src_base = nw * 1024 + lane;
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
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[16])) : "r"(smk_part_addr + (unsigned int)((src_base + 512) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[17])) : "r"(smk_part_addr + (unsigned int)((src_base + 544) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[18])) : "r"(smk_part_addr + (unsigned int)((src_base + 576) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[19])) : "r"(smk_part_addr + (unsigned int)((src_base + 608) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[20])) : "r"(smk_part_addr + (unsigned int)((src_base + 640) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[21])) : "r"(smk_part_addr + (unsigned int)((src_base + 672) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[22])) : "r"(smk_part_addr + (unsigned int)((src_base + 704) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[23])) : "r"(smk_part_addr + (unsigned int)((src_base + 736) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[24])) : "r"(smk_part_addr + (unsigned int)((src_base + 768) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[25])) : "r"(smk_part_addr + (unsigned int)((src_base + 800) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[26])) : "r"(smk_part_addr + (unsigned int)((src_base + 832) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[27])) : "r"(smk_part_addr + (unsigned int)((src_base + 864) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[28])) : "r"(smk_part_addr + (unsigned int)((src_base + 896) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[29])) : "r"(smk_part_addr + (unsigned int)((src_base + 928) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[30])) : "r"(smk_part_addr + (unsigned int)((src_base + 960) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[31])) : "r"(smk_part_addr + (unsigned int)((src_base + 992) * 4)));
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
        acc[16] = acc[16] + part[16];
        acc[17] = acc[17] + part[17];
        acc[18] = acc[18] + part[18];
        acc[19] = acc[19] + part[19];
        acc[20] = acc[20] + part[20];
        acc[21] = acc[21] + part[21];
        acc[22] = acc[22] + part[22];
        acc[23] = acc[23] + part[23];
        acc[24] = acc[24] + part[24];
        acc[25] = acc[25] + part[25];
        acc[26] = acc[26] + part[26];
        acc[27] = acc[27] + part[27];
        acc[28] = acc[28] + part[28];
        acc[29] = acc[29] + part[29];
        acc[30] = acc[30] + part[30];
        acc[31] = acc[31] + part[31];
        int src_base_0 = (2 + nw) * 1024 + lane;
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
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[16])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 512) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[17])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 544) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[18])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 576) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[19])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 608) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[20])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 640) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[21])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 672) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[22])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 704) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[23])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 736) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[24])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 768) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[25])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 800) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[26])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 832) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[27])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 864) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[28])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 896) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[29])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 928) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[30])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 960) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[31])) : "r"(smk_part_addr + (unsigned int)((src_base_0 + 992) * 4)));
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
        acc[16] = acc[16] + part[16];
        acc[17] = acc[17] + part[17];
        acc[18] = acc[18] + part[18];
        acc[19] = acc[19] + part[19];
        acc[20] = acc[20] + part[20];
        acc[21] = acc[21] + part[21];
        acc[22] = acc[22] + part[22];
        acc[23] = acc[23] + part[23];
        acc[24] = acc[24] + part[24];
        acc[25] = acc[25] + part[25];
        acc[26] = acc[26] + part[26];
        acc[27] = acc[27] + part[27];
        acc[28] = acc[28] + part[28];
        acc[29] = acc[29] + part[29];
        acc[30] = acc[30] + part[30];
        acc[31] = acc[31] + part[31];
        int src_base_1 = (4 + nw) * 1024 + lane;
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
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[16])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 512) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[17])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 544) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[18])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 576) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[19])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 608) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[20])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 640) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[21])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 672) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[22])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 704) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[23])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 736) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[24])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 768) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[25])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 800) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[26])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 832) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[27])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 864) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[28])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 896) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[29])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 928) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[30])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 960) * 4)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&part[31])) : "r"(smk_part_addr + (unsigned int)((src_base_1 + 992) * 4)));
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
        acc[16] = acc[16] + part[16];
        acc[17] = acc[17] + part[17];
        acc[18] = acc[18] + part[18];
        acc[19] = acc[19] + part[19];
        acc[20] = acc[20] + part[20];
        acc[21] = acc[21] + part[21];
        acc[22] = acc[22] + part[22];
        acc[23] = acc[23] + part[23];
        acc[24] = acc[24] + part[24];
        acc[25] = acc[25] + part[25];
        acc[26] = acc[26] + part[26];
        acc[27] = acc[27] + part[27];
        acc[28] = acc[28] + part[28];
        acc[29] = acc[29] + part[29];
        acc[30] = acc[30] + part[30];
        acc[31] = acc[31] + part[31];
        long long plane_elems = (long long)M * (long long)N;
        long long plane = (long long)blockIdx.z * plane_elems;
        int col = off_n + nw * 8 + lane_r * 2;
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
        int row_2 = lane_g + 8;
        if (row_2 < M) {
            long long base_1 = plane + (long long)row_2 * (long long)N + (long long)col;
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
        int row_3 = lane_g + 16;
        if (row_3 < M) {
            long long base_2 = plane + (long long)row_3 * (long long)N + (long long)col;
            {
                float2 _v2 = make_float2(acc[16 + 0], acc[16 + 1]);
                *reinterpret_cast<float2*>((partials + base_2) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[20 + 0], acc[20 + 1]);
                *reinterpret_cast<float2*>((partials + (base_2 + 16)) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[24 + 0], acc[24 + 1]);
                *reinterpret_cast<float2*>((partials + (base_2 + 32)) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[28 + 0], acc[28 + 1]);
                *reinterpret_cast<float2*>((partials + (base_2 + 48)) + 0) = _v2;
            }
        }
        int row_4 = lane_g + 24;
        if (row_4 < M) {
            long long base_3 = plane + (long long)row_4 * (long long)N + (long long)col;
            {
                float2 _v2 = make_float2(acc[18 + 0], acc[18 + 1]);
                *reinterpret_cast<float2*>((partials + base_3) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[22 + 0], acc[22 + 1]);
                *reinterpret_cast<float2*>((partials + (base_3 + 16)) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[26 + 0], acc[26 + 1]);
                *reinterpret_cast<float2*>((partials + (base_3 + 32)) + 0) = _v2;
            }
            {
                float2 _v2 = make_float2(acc[30 + 0], acc[30 + 1]);
                *reinterpret_cast<float2*>((partials + (base_3 + 48)) + 0) = _v2;
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        int old = 0;
        if (lane == 0) {
            unsigned int old_u;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                : "=r"(old_u) : "l"(&completion_u32[blockIdx.x]), "r"(static_cast<uint32_t>(1)) : "memory");
            old = (int)old_u;
            smk_flag[0] = old;
        }
    }
    __syncthreads();
    int flag[1];
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&flag[0])) : "r"(smk_flag_addr));
    if (flag[0] == 5) {
        float alpha_value = 1.0f;
        long long plane_elems_1 = (long long)M * (long long)N;
        int tid_0 = tid;
        int chunk = tid_0;
        int row_1 = chunk >> 4;
        if (row_1 < M) {
            int col_1 = off_n + (chunk & 15) * 4;
            long long base_4 = (long long)row_1 * (long long)N + (long long)col_1;
            float fin[4];
            float _vec_load_0[4];
            {
                unsigned _v4_0_0;
                unsigned _v4_0_1;
                unsigned _v4_0_2;
                unsigned _v4_0_3;
                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3) : "l"((const void*)(partials + (base_4))) : "memory");
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
                    : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3) : "l"((const void*)(partials + (base_4 + plane_elems_1))) : "memory");
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
                    : "=r"(_v4_2_0), "=r"(_v4_2_1), "=r"(_v4_2_2), "=r"(_v4_2_3) : "l"((const void*)(partials + (base_4 + 2 * plane_elems_1))) : "memory");
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
                    : "=r"(_v4_3_0), "=r"(_v4_3_1), "=r"(_v4_3_2), "=r"(_v4_3_3) : "l"((const void*)(partials + (base_4 + 3 * plane_elems_1))) : "memory");
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
                    : "=r"(_v4_4_0), "=r"(_v4_4_1), "=r"(_v4_4_2), "=r"(_v4_4_3) : "l"((const void*)(partials + (base_4 + 4 * plane_elems_1))) : "memory");
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
                    : "=r"(_v4_5_0), "=r"(_v4_5_1), "=r"(_v4_5_2), "=r"(_v4_5_3) : "l"((const void*)(partials + (base_4 + 5 * plane_elems_1))) : "memory");
                _vec_load_5[0 + 0] = __uint_as_float(_v4_5_0);
                _vec_load_5[0 + 1] = __uint_as_float(_v4_5_1);
                _vec_load_5[0 + 2] = __uint_as_float(_v4_5_2);
                _vec_load_5[0 + 3] = __uint_as_float(_v4_5_3);
            }
            fin[0] = fin[0] + _vec_load_5[0];
            fin[1] = fin[1] + _vec_load_5[1];
            fin[2] = fin[2] + _vec_load_5[2];
            fin[3] = fin[3] + _vec_load_5[3];
            float _scaled_0[4];
            const float2 _scale2_6 = {alpha_value, alpha_value};
            #pragma unroll
            for (int _ls = 0; _ls < 2; _ls++)
                reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(fin)[_ls], _scale2_6);
            unsigned int _packed_0[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 0], _scaled_0[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 2], _scaled_0[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(C + base_4)[0] = reinterpret_cast<int2*>(_packed_0)[0];
        }
        int chunk_1 = tid_0 + 256;
        int row_2_1 = chunk_1 >> 4;
        if (row_2_1 < M) {
            int col_2 = off_n + (chunk_1 & 15) * 4;
            long long base_5 = (long long)row_2_1 * (long long)N + (long long)col_2;
            float fin_1[4];
            float _vec_load_6[4];
            {
                unsigned _v4_7_0;
                unsigned _v4_7_1;
                unsigned _v4_7_2;
                unsigned _v4_7_3;
                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_7_0), "=r"(_v4_7_1), "=r"(_v4_7_2), "=r"(_v4_7_3) : "l"((const void*)(partials + (base_5))) : "memory");
                _vec_load_6[0 + 0] = __uint_as_float(_v4_7_0);
                _vec_load_6[0 + 1] = __uint_as_float(_v4_7_1);
                _vec_load_6[0 + 2] = __uint_as_float(_v4_7_2);
                _vec_load_6[0 + 3] = __uint_as_float(_v4_7_3);
            }
            fin_1[0] = _vec_load_6[0];
            fin_1[1] = _vec_load_6[1];
            fin_1[2] = _vec_load_6[2];
            fin_1[3] = _vec_load_6[3];
            float _vec_load_7[4];
            {
                unsigned _v4_8_0;
                unsigned _v4_8_1;
                unsigned _v4_8_2;
                unsigned _v4_8_3;
                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_8_0), "=r"(_v4_8_1), "=r"(_v4_8_2), "=r"(_v4_8_3) : "l"((const void*)(partials + (base_5 + plane_elems_1))) : "memory");
                _vec_load_7[0 + 0] = __uint_as_float(_v4_8_0);
                _vec_load_7[0 + 1] = __uint_as_float(_v4_8_1);
                _vec_load_7[0 + 2] = __uint_as_float(_v4_8_2);
                _vec_load_7[0 + 3] = __uint_as_float(_v4_8_3);
            }
            fin_1[0] = fin_1[0] + _vec_load_7[0];
            fin_1[1] = fin_1[1] + _vec_load_7[1];
            fin_1[2] = fin_1[2] + _vec_load_7[2];
            fin_1[3] = fin_1[3] + _vec_load_7[3];
            float _vec_load_8[4];
            {
                unsigned _v4_9_0;
                unsigned _v4_9_1;
                unsigned _v4_9_2;
                unsigned _v4_9_3;
                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_9_0), "=r"(_v4_9_1), "=r"(_v4_9_2), "=r"(_v4_9_3) : "l"((const void*)(partials + (base_5 + 2 * plane_elems_1))) : "memory");
                _vec_load_8[0 + 0] = __uint_as_float(_v4_9_0);
                _vec_load_8[0 + 1] = __uint_as_float(_v4_9_1);
                _vec_load_8[0 + 2] = __uint_as_float(_v4_9_2);
                _vec_load_8[0 + 3] = __uint_as_float(_v4_9_3);
            }
            fin_1[0] = fin_1[0] + _vec_load_8[0];
            fin_1[1] = fin_1[1] + _vec_load_8[1];
            fin_1[2] = fin_1[2] + _vec_load_8[2];
            fin_1[3] = fin_1[3] + _vec_load_8[3];
            float _vec_load_9[4];
            {
                unsigned _v4_10_0;
                unsigned _v4_10_1;
                unsigned _v4_10_2;
                unsigned _v4_10_3;
                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_v4_10_0), "=r"(_v4_10_1), "=r"(_v4_10_2), "=r"(_v4_10_3) : "l"((const void*)(partials + (base_5 + 3 * plane_elems_1))) : "memory");
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
                    : "=r"(_v4_11_0), "=r"(_v4_11_1), "=r"(_v4_11_2), "=r"(_v4_11_3) : "l"((const void*)(partials + (base_5 + 4 * plane_elems_1))) : "memory");
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
                    : "=r"(_v4_12_0), "=r"(_v4_12_1), "=r"(_v4_12_2), "=r"(_v4_12_3) : "l"((const void*)(partials + (base_5 + 5 * plane_elems_1))) : "memory");
                _vec_load_11[0 + 0] = __uint_as_float(_v4_12_0);
                _vec_load_11[0 + 1] = __uint_as_float(_v4_12_1);
                _vec_load_11[0 + 2] = __uint_as_float(_v4_12_2);
                _vec_load_11[0 + 3] = __uint_as_float(_v4_12_3);
            }
            fin_1[0] = fin_1[0] + _vec_load_11[0];
            fin_1[1] = fin_1[1] + _vec_load_11[1];
            fin_1[2] = fin_1[2] + _vec_load_11[2];
            fin_1[3] = fin_1[3] + _vec_load_11[3];
            float _scaled_1[4];
            const float2 _scale2_13 = {alpha_value, alpha_value};
            #pragma unroll
            for (int _ls = 0; _ls < 2; _ls++)
                reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(fin_1)[_ls], _scale2_13);
            unsigned int _packed_1[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 0], _scaled_1[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 2], _scaled_1[_lp*2+1 + 2]));
                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(C + base_5)[0] = reinterpret_cast<int2*>(_packed_1)[0];
        }
        if (tid == 0) {
            completion_u32[blockIdx.x] = 0;
        }
    }
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
