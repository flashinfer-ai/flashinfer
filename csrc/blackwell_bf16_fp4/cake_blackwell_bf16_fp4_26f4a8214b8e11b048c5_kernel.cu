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
#define SMEM_SMK_A_STRIDE 17408
#define SMEM_SMK_B_OFF 9216
#define SMEM_SMK_B_STAGE_BYTES 8192
#define SMEM_SMK_B_STRIDE 17408
#define SMEM_SMK_SF_OFF 17408
#define SMEM_SMK_SF_STAGE_BYTES 1024
#define SMEM_SMK_SF_STRIDE 17408
#define SMEM_SMK_SFX_OFF 18432
#define SMEM_SMK_SFX_STAGE_BYTES 4096
#define SMEM_SMK_SFX_STRIDE 4096
#define SMEM_SMK_PART_OFF 22528
#define SMEM_SMK_PART_STAGE_BYTES 12288
#define SMEM_SMK_PART_STRIDE 12288
#define SMEM_SMK_MAIL_OFF 34816
#define SMEM_SMK_MAIL_STAGE_BYTES 12288
#define SMEM_SMK_MAIL_STRIDE 12288
#define SMEM_TOTAL 47104
#define THREADS 256
#define HAS_ALPHA 0
#define ENABLE_PDL 1
#define LAUNCH_MIN_BLOCKS 1

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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(1,1,4) void
kernel_cake_blackwell_bf16_fp4_26f4a8214b8e11b048c5(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
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
    #define mail_bar_addr (mbar_base + 8)
    #define sf_full_addr (mbar_base + 16)

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
    int* smk_b = reinterpret_cast<int*>(smem_raw + 9216);
    const int smk_b_addr = smem + 9216;
    uint8_t* smk_sf = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smk_sf_addr = smem + 17408;
    unsigned int* smk_sfx = reinterpret_cast<unsigned int*>(smem_raw + 18432);
    const int smk_sfx_addr = smem + 18432;
    float* smk_part = reinterpret_cast<float*>(smem_raw + 22528);
    const int smk_part_addr = smem + 22528;
    float* smk_mail = reinterpret_cast<float*>(smem_raw + 34816);
    const int smk_mail_addr = smem + 34816;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // mail_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // sf_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 8, 12288);
        }
    }

    // === Task calls (declaration order) ===
    int off_n = blockIdx.x * 64;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(sf_full_addr, 1024);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_sf_addr), "l"((&B_descale)), "r"(off_n), "r"(blockIdx.z * 16),
                   "r"(sf_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            mbarrier_arrive_expect_tx(full_addr, 16384);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(smk_b_addr), "l"((&B)), "r"(off_n * 2), "r"(blockIdx.z * 16),
                   "r"(full_addr), "l"(0x12F0000000000000ULL) : "memory");
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
                :: "r"(smk_a_addr), "l"((&A)), "r"(0), "r"(0), "r"(blockIdx.z * 4),
                   "r"(full_addr), "l"(0x14F0000000000000ULL) : "memory");
        }
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
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
        : "r"((smk_a_addr + (unsigned int)((64 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (64 + pair * 16 + a_col) % 64 * 2 ^ ((64 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (64 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
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
        : "r"((smk_a_addr + (unsigned int)((128 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (128 + pair * 16 + a_col) % 64 * 2 ^ ((128 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (128 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
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
        : "r"((smk_a_addr + (unsigned int)((192 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (192 + pair * 16 + a_col) % 64 * 2 ^ ((192 + pair * 16 + a_col) / 64 * 2048 + a_row * 128 + (192 + pair * 16 + a_col) % 64 * 2 >> 7 & 7) << 4)))
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
    float part[16];
    int rec_base = ((pair - 1) * 2 + nw) * 512 + lane;
    if (pair > 0) {
        unsigned int vec_dst = smk_part_addr + (unsigned int)(((pair - 1) * 2 + nw) * 2048 + lane * 16);
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(vec_dst), "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]) : "memory");
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(vec_dst + 512), "f"(acc[4]), "f"(acc[5]), "f"(acc[6]), "f"(acc[7]) : "memory");
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(vec_dst + 1024), "f"(acc[8]), "f"(acc[9]), "f"(acc[10]), "f"(acc[11]) : "memory");
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(vec_dst + 1536), "f"(acc[12]), "f"(acc[13]), "f"(acc[14]), "f"(acc[15]) : "memory");
    }
    __syncthreads();
    if (pair < 1) {
        unsigned int vec_src_0 = smk_part_addr + (unsigned int)(nw * 2048 + lane * 16);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
            : "r"(vec_src_0));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
            : "r"(vec_src_0 + 512));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[8])), "=r"(*reinterpret_cast<uint32_t*>(&part[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(8) + 3]))
            : "r"(vec_src_0 + 1024));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[12])), "=r"(*reinterpret_cast<uint32_t*>(&part[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(12) + 3]))
            : "r"(vec_src_0 + 1536));
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
        unsigned int vec_src_1 = smk_part_addr + (unsigned int)((2 + nw) * 2048 + lane * 16);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
            : "r"(vec_src_1));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
            : "r"(vec_src_1 + 512));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[8])), "=r"(*reinterpret_cast<uint32_t*>(&part[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(8) + 3]))
            : "r"(vec_src_1 + 1024));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[12])), "=r"(*reinterpret_cast<uint32_t*>(&part[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(12) + 3]))
            : "r"(vec_src_1 + 1536));
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
        unsigned int vec_src_2 = smk_part_addr + (unsigned int)((4 + nw) * 2048 + lane * 16);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
            : "r"(vec_src_2));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(4) + 3]))
            : "r"(vec_src_2 + 512));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[8])), "=r"(*reinterpret_cast<uint32_t*>(&part[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(8) + 3]))
            : "r"(vec_src_2 + 1024));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[12])), "=r"(*reinterpret_cast<uint32_t*>(&part[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(12) + 3]))
            : "r"(vec_src_2 + 1536));
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
        int rec_off = nw * 2048 + lane * 16;
        if (cta_rank > 0) {
            int mail_dst = smk_mail_addr + (unsigned int)((cta_rank - 1) * 4096 + rec_off);
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(mail_dst), "r"(0));
            uint32_t _mbar_peer_0;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_0) : "r"(mail_bar_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])), "r"(_mbar_peer_0) : "memory");
            uint32_t _mbar_peer_1;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_1) : "r"(mail_bar_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 512), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])), "r"(_mbar_peer_1) : "memory");
            uint32_t _mbar_peer_2;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_2) : "r"(mail_bar_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 1024), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])), "r"(_mbar_peer_2) : "memory");
            uint32_t _mbar_peer_3;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_3) : "r"(mail_bar_addr), "r"(0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 1536), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])), "r"(_mbar_peer_3) : "memory");
        } else {
            mbarrier_wait_cluster_hint(mail_bar_addr, 0, 10000000);
            float part_0[16];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)rec_off));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(4) + 3]))
                : "r"(smk_mail_addr + (unsigned int)(512 + rec_off)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(8) + 3]))
                : "r"(smk_mail_addr + (unsigned int)(1024 + rec_off)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(12) + 3]))
                : "r"(smk_mail_addr + (unsigned int)(1536 + rec_off)));
            acc[0] = acc[0] + part_0[0];
            acc[1] = acc[1] + part_0[1];
            acc[2] = acc[2] + part_0[2];
            acc[3] = acc[3] + part_0[3];
            acc[4] = acc[4] + part_0[4];
            acc[5] = acc[5] + part_0[5];
            acc[6] = acc[6] + part_0[6];
            acc[7] = acc[7] + part_0[7];
            acc[8] = acc[8] + part_0[8];
            acc[9] = acc[9] + part_0[9];
            acc[10] = acc[10] + part_0[10];
            acc[11] = acc[11] + part_0[11];
            acc[12] = acc[12] + part_0[12];
            acc[13] = acc[13] + part_0[13];
            acc[14] = acc[14] + part_0[14];
            acc[15] = acc[15] + part_0[15];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)(4096 + rec_off)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(4) + 3]))
                : "r"(smk_mail_addr + (unsigned int)(4608 + rec_off)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(8) + 3]))
                : "r"(smk_mail_addr + (unsigned int)(5120 + rec_off)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(12) + 3]))
                : "r"(smk_mail_addr + (unsigned int)(5632 + rec_off)));
            acc[0] = acc[0] + part_0[0];
            acc[1] = acc[1] + part_0[1];
            acc[2] = acc[2] + part_0[2];
            acc[3] = acc[3] + part_0[3];
            acc[4] = acc[4] + part_0[4];
            acc[5] = acc[5] + part_0[5];
            acc[6] = acc[6] + part_0[6];
            acc[7] = acc[7] + part_0[7];
            acc[8] = acc[8] + part_0[8];
            acc[9] = acc[9] + part_0[9];
            acc[10] = acc[10] + part_0[10];
            acc[11] = acc[11] + part_0[11];
            acc[12] = acc[12] + part_0[12];
            acc[13] = acc[13] + part_0[13];
            acc[14] = acc[14] + part_0[14];
            acc[15] = acc[15] + part_0[15];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(0) + 3]))
                : "r"(smk_mail_addr + (unsigned int)(8192 + rec_off)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(4) + 3]))
                : "r"(smk_mail_addr + (unsigned int)(8704 + rec_off)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(8) + 3]))
                : "r"(smk_mail_addr + (unsigned int)(9216 + rec_off)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&part_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_0[(12) + 3]))
                : "r"(smk_mail_addr + (unsigned int)(9728 + rec_off)));
            acc[0] = acc[0] + part_0[0];
            acc[1] = acc[1] + part_0[1];
            acc[2] = acc[2] + part_0[2];
            acc[3] = acc[3] + part_0[3];
            acc[4] = acc[4] + part_0[4];
            acc[5] = acc[5] + part_0[5];
            acc[6] = acc[6] + part_0[6];
            acc[7] = acc[7] + part_0[7];
            acc[8] = acc[8] + part_0[8];
            acc[9] = acc[9] + part_0[9];
            acc[10] = acc[10] + part_0[10];
            acc[11] = acc[11] + part_0[11];
            acc[12] = acc[12] + part_0[12];
            acc[13] = acc[13] + part_0[13];
            acc[14] = acc[14] + part_0[14];
            acc[15] = acc[15] + part_0[15];
            float alpha_value = 1.0f;
            int row0 = lane_g;
            int row1 = lane_g + 8;
            int col = off_n + nw * 8 + lane_r * 2;
            if (row0 < M) {
                long long base0 = (long long)row0 * (long long)N + (long long)col;
                float _scaled_0[2];
                const float2 _scale2_4 = {alpha_value, alpha_value};
                #pragma unroll
                for (int _ls = 0; _ls < 1; _ls++)
                    reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(acc)[_ls], _scale2_4);
                unsigned int _packed_0[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 0], _scaled_0[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(C + base0)[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
                float _scaled_1[2];
                const float2 _scale2_5 = {alpha_value, alpha_value};
                #pragma unroll
                for (int _ls = 0; _ls < 1; _ls++)
                    reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((acc + 4))[_ls], _scale2_5);
                unsigned int _packed_1[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 0], _scaled_1[_lp*2+1 + 0]));
                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(C + (base0 + 16))[0] = reinterpret_cast<unsigned*>(_packed_1)[0];
                float _scaled_2[2];
                const float2 _scale2_6 = {alpha_value, alpha_value};
                #pragma unroll
                for (int _ls = 0; _ls < 1; _ls++)
                    reinterpret_cast<float2*>(_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((acc + 8))[_ls], _scale2_6);
                unsigned int _packed_2[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 0], _scaled_2[_lp*2+1 + 0]));
                    _packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(C + (base0 + 32))[0] = reinterpret_cast<unsigned*>(_packed_2)[0];
                float _scaled_3[2];
                const float2 _scale2_7 = {alpha_value, alpha_value};
                #pragma unroll
                for (int _ls = 0; _ls < 1; _ls++)
                    reinterpret_cast<float2*>(_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((acc + 12))[_ls], _scale2_7);
                unsigned int _packed_3[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 0], _scaled_3[_lp*2+1 + 0]));
                    _packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(C + (base0 + 48))[0] = reinterpret_cast<unsigned*>(_packed_3)[0];
            }
            if (row1 < M) {
                long long base1 = (long long)row1 * (long long)N + (long long)col;
                float _scaled_4[2];
                const float2 _scale2_8 = {alpha_value, alpha_value};
                #pragma unroll
                for (int _ls = 0; _ls < 1; _ls++)
                    reinterpret_cast<float2*>(_scaled_4)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((acc + 2))[_ls], _scale2_8);
                unsigned int _packed_4[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4[_lp*2 + 0], _scaled_4[_lp*2+1 + 0]));
                    _packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(C + base1)[0] = reinterpret_cast<unsigned*>(_packed_4)[0];
                float _scaled_5[2];
                const float2 _scale2_9 = {alpha_value, alpha_value};
                #pragma unroll
                for (int _ls = 0; _ls < 1; _ls++)
                    reinterpret_cast<float2*>(_scaled_5)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((acc + 6))[_ls], _scale2_9);
                unsigned int _packed_5[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 0], _scaled_5[_lp*2+1 + 0]));
                    _packed_5[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(C + (base1 + 16))[0] = reinterpret_cast<unsigned*>(_packed_5)[0];
                float _scaled_6[2];
                const float2 _scale2_10 = {alpha_value, alpha_value};
                #pragma unroll
                for (int _ls = 0; _ls < 1; _ls++)
                    reinterpret_cast<float2*>(_scaled_6)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((acc + 10))[_ls], _scale2_10);
                unsigned int _packed_6[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 0], _scaled_6[_lp*2+1 + 0]));
                    _packed_6[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(C + (base1 + 32))[0] = reinterpret_cast<unsigned*>(_packed_6)[0];
                float _scaled_7[2];
                const float2 _scale2_11 = {alpha_value, alpha_value};
                #pragma unroll
                for (int _ls = 0; _ls < 1; _ls++)
                    reinterpret_cast<float2*>(_scaled_7)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((acc + 14))[_ls], _scale2_11);
                unsigned int _packed_7[1];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 0], _scaled_7[_lp*2+1 + 0]));
                    _packed_7[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<unsigned*>(C + (base1 + 48))[0] = reinterpret_cast<unsigned*>(_packed_7)[0];
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
