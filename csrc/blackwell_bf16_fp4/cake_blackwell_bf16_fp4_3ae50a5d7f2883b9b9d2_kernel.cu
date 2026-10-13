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
#define NUM_PIPE_STAGES 16
#define SMEM_WARP_MMA_A_OFF 1024
#define SMEM_WARP_MMA_A_STAGE_BYTES 4096
#define SMEM_WARP_MMA_A_STRIDE 9216
#define SMEM_WARP_MMA_B_PACKED_OFF 5120
#define SMEM_WARP_MMA_B_PACKED_STAGE_BYTES 4096
#define SMEM_WARP_MMA_B_PACKED_STRIDE 9216
#define SMEM_WARP_MMA_B_SCALE_OFF 9216
#define SMEM_WARP_MMA_B_SCALE_STAGE_BYTES 512
#define SMEM_WARP_MMA_B_SCALE_STRIDE 9216
#define SMEM_WARP_MMA_C_OFF 148480
#define SMEM_WARP_MMA_C_STAGE_BYTES 2048
#define SMEM_WARP_MMA_C_STRIDE 2048
#define SMEM_TOTAL 150528
#define THREADS 96
#define HAS_ALPHA 0
#define ENABLE_PDL 1
#define FLAT_GRID 0

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


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(96) void
kernel_cake_blackwell_bf16_fp4_3ae50a5d7f2883b9b9d2(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap B_descale, float* __restrict__ alpha, const __grid_constant__ CUtensorMap C, int M, int N, int K)
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
    #define ab_full_addr (mbar_base + 0)
    #define ab_free_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* warp_mma_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int warp_mma_a_addr = smem + 1024;
    int* warp_mma_b_packed = reinterpret_cast<int*>(smem_raw + 5120);
    const int warp_mma_b_packed_addr = smem + 5120;
    uint8_t* warp_mma_b_scale = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int warp_mma_b_scale_addr = smem + 9216;
    __nv_bfloat16* warp_mma_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int warp_mma_c_addr = smem + 148480;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // ab_full: 16 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // ab_free: 16 barriers, init_count=2
            mbarrier_init(smem + 128, 2);
            mbarrier_init(smem + 136, 2);
            mbarrier_init(smem + 144, 2);
            mbarrier_init(smem + 152, 2);
            mbarrier_init(smem + 160, 2);
            mbarrier_init(smem + 168, 2);
            mbarrier_init(smem + 176, 2);
            mbarrier_init(smem + 184, 2);
            mbarrier_init(smem + 192, 2);
            mbarrier_init(smem + 200, 2);
            mbarrier_init(smem + 208, 2);
            mbarrier_init(smem + 216, 2);
            mbarrier_init(smem + 224, 2);
            mbarrier_init(smem + 232, 2);
            mbarrier_init(smem + 240, 2);
            mbarrier_init(smem + 248, 2);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: compute ----
    if (warp <= 1) {
        { // compute_main
            int grid_n = (N + 64 - 1) / 64;
            int grid_m = (M + 16 - 1) / 16;
            int total_tiles = grid_m * grid_n;
            unsigned int compute_stage = 0;
            unsigned int a_frag[4];
            unsigned int a_frag_hi[4];
            unsigned int raw[2];
            unsigned int scale_word[1];
            float acc0[4];
            float acc1[4];
            float acc2[4];
            float acc3[4];
            float acc0_hi[4];
            float acc1_hi[4];
            float acc2_hi[4];
            float acc3_hi[4];
            float epi01[8];
            float epi23[8];
            float epi01_hi[8];
            float epi23_hi[8];
            unsigned int packed01[4];
            unsigned int packed23[4];
            unsigned int packed01_hi[4];
            unsigned int packed23_hi[4];
            int warp_id_in_role = (warp - 0);
            int m_warp = 0;
            int n_warp = warp_id_in_role;
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (unsigned int work = blockIdx.x; work < total_tiles; work += gridDim.x) {
                int tile_m = work / (unsigned int)grid_n;
                int tile_n = work - (unsigned int)(tile_m * grid_n);
                int off_m = tile_m * 16;
                int off_n = tile_n * 64;
                acc0[0] = 0.0f;
                acc0[1] = 0.0f;
                acc0[2] = 0.0f;
                acc0[3] = 0.0f;
                acc1[0] = 0.0f;
                acc1[1] = 0.0f;
                acc1[2] = 0.0f;
                acc1[3] = 0.0f;
                acc2[0] = 0.0f;
                acc2[1] = 0.0f;
                acc2[2] = 0.0f;
                acc2[3] = 0.0f;
                acc3[0] = 0.0f;
                acc3[1] = 0.0f;
                acc3[2] = 0.0f;
                acc3[3] = 0.0f;
                int k_tiles = (K + 128 - 1) / 128;
                #pragma unroll 1
                for (int kt = 0; kt < k_tiles; kt++) {
                    mbarrier_wait(ab_full_addr + (compute_stage) * 8, _phase_ab_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int a_base = warp_mma_a_addr + compute_stage * 9216;
                    int b_base = warp_mma_b_packed_addr + compute_stage * 9216;
                    int sf_base = warp_mma_b_scale_addr + compute_stage * 9216;
                    int tc_col = lane / 4;
                    int base_n = n_warp * 8 + tc_col;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"((a_base + ((m_warp * 16 + lane % 16) * 128 + lane / 16 * 8 * 2 ^ ((m_warp * 16 + lane % 16) * 128 + lane / 16 * 8 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    uint32_t _mma_sync_m16n8k16_a_f16_0[4];
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_0[0]) : "r"(a_frag[0]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_0[1]) : "r"(a_frag[1]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_0[2]) : "r"(a_frag[2]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_0[3]) : "r"(a_frag[3]));
                    int u32_pos = n_warp * 64 + lane * 2;
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
                        : "r"(b_base + u32_pos * 4));
                    int sf_linear = base_n;
                    uint8_t scale_byte = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear / 4 * 4));
                        scale_byte = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear & 3) * 8) & 255);
                    }
                    unsigned int packed_word = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_0;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_0) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_1;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_1) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_0[2];
                    _mma_sync_m16n8k16_b_0[0] = _fp4_dequant_x2_0;
                    _mma_sync_m16n8k16_b_0[1] = _fp4_dequant_x2_1;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_0[0]), "r"(_mma_sync_m16n8k16_a_f16_0[1]), "r"(_mma_sync_m16n8k16_a_f16_0[2]), "r"(_mma_sync_m16n8k16_a_f16_0[3]), "r"(_mma_sync_m16n8k16_b_0[0]), "r"(_mma_sync_m16n8k16_b_0[1]));
                    }
                    int sf_linear_0 = base_n + 16;
                    uint8_t scale_byte_1 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_0 / 4 * 4));
                        scale_byte_1 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_0 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_2 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_2;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_2 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_1)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_3;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_2 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_1)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_1[2];
                    _mma_sync_m16n8k16_b_1[0] = _fp4_dequant_x2_2;
                    _mma_sync_m16n8k16_b_1[1] = _fp4_dequant_x2_3;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_0[0]), "r"(_mma_sync_m16n8k16_a_f16_0[1]), "r"(_mma_sync_m16n8k16_a_f16_0[2]), "r"(_mma_sync_m16n8k16_a_f16_0[3]), "r"(_mma_sync_m16n8k16_b_1[0]), "r"(_mma_sync_m16n8k16_b_1[1]));
                    }
                    int sf_linear_3 = base_n + 32;
                    uint8_t scale_byte_4 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_3 / 4 * 4));
                        scale_byte_4 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_3 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_5 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_4;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_5 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_4)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_5;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_5 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_4)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_2[2];
                    _mma_sync_m16n8k16_b_2[0] = _fp4_dequant_x2_4;
                    _mma_sync_m16n8k16_b_2[1] = _fp4_dequant_x2_5;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc2[0]), "+f"(acc2[1]), "+f"(acc2[2]), "+f"(acc2[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_0[0]), "r"(_mma_sync_m16n8k16_a_f16_0[1]), "r"(_mma_sync_m16n8k16_a_f16_0[2]), "r"(_mma_sync_m16n8k16_a_f16_0[3]), "r"(_mma_sync_m16n8k16_b_2[0]), "r"(_mma_sync_m16n8k16_b_2[1]));
                    }
                    int sf_linear_6 = base_n + 48;
                    uint8_t scale_byte_7 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_6 / 4 * 4));
                        scale_byte_7 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_6 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_8 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_6;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_8 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_7)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_7;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_8 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_7)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_3[2];
                    _mma_sync_m16n8k16_b_3[0] = _fp4_dequant_x2_6;
                    _mma_sync_m16n8k16_b_3[1] = _fp4_dequant_x2_7;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc3[0]), "+f"(acc3[1]), "+f"(acc3[2]), "+f"(acc3[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_0[0]), "r"(_mma_sync_m16n8k16_a_f16_0[1]), "r"(_mma_sync_m16n8k16_a_f16_0[2]), "r"(_mma_sync_m16n8k16_a_f16_0[3]), "r"(_mma_sync_m16n8k16_b_3[0]), "r"(_mma_sync_m16n8k16_b_3[1]));
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"((a_base + ((m_warp * 16 + lane % 16) * 128 + (16 + lane / 16 * 8) * 2 ^ ((m_warp * 16 + lane % 16) * 128 + (16 + lane / 16 * 8) * 2 >> 7 & 7) << 4)))
                        : "memory");
                    uint32_t _mma_sync_m16n8k16_a_f16_1[4];
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_1[0]) : "r"(a_frag[0]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_1[1]) : "r"(a_frag[1]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_1[2]) : "r"(a_frag[2]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_1[3]) : "r"(a_frag[3]));
                    int u32_pos_9 = n_warp * 64 + lane * 2;
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
                        : "r"(b_base + (128 + u32_pos_9) * 4));
                    int sf_linear_10 = 64 + base_n;
                    uint8_t scale_byte_11 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_10 / 4 * 4));
                        scale_byte_11 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_10 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_12 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_8;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_12 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_11)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_8) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_9;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_12 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_11)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_9) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_4[2];
                    _mma_sync_m16n8k16_b_4[0] = _fp4_dequant_x2_8;
                    _mma_sync_m16n8k16_b_4[1] = _fp4_dequant_x2_9;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_1[0]), "r"(_mma_sync_m16n8k16_a_f16_1[1]), "r"(_mma_sync_m16n8k16_a_f16_1[2]), "r"(_mma_sync_m16n8k16_a_f16_1[3]), "r"(_mma_sync_m16n8k16_b_4[0]), "r"(_mma_sync_m16n8k16_b_4[1]));
                    }
                    int sf_linear_13 = 64 + base_n + 16;
                    uint8_t scale_byte_14 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_13 / 4 * 4));
                        scale_byte_14 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_13 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_15 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_10;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_15 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_14)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_10) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_11;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_15 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_14)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_11) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_5[2];
                    _mma_sync_m16n8k16_b_5[0] = _fp4_dequant_x2_10;
                    _mma_sync_m16n8k16_b_5[1] = _fp4_dequant_x2_11;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_1[0]), "r"(_mma_sync_m16n8k16_a_f16_1[1]), "r"(_mma_sync_m16n8k16_a_f16_1[2]), "r"(_mma_sync_m16n8k16_a_f16_1[3]), "r"(_mma_sync_m16n8k16_b_5[0]), "r"(_mma_sync_m16n8k16_b_5[1]));
                    }
                    int sf_linear_16 = 64 + base_n + 32;
                    uint8_t scale_byte_17 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_16 / 4 * 4));
                        scale_byte_17 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_16 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_18 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_12;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_18 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_17)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_12) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_13;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_18 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_17)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_13) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_6[2];
                    _mma_sync_m16n8k16_b_6[0] = _fp4_dequant_x2_12;
                    _mma_sync_m16n8k16_b_6[1] = _fp4_dequant_x2_13;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc2[0]), "+f"(acc2[1]), "+f"(acc2[2]), "+f"(acc2[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_1[0]), "r"(_mma_sync_m16n8k16_a_f16_1[1]), "r"(_mma_sync_m16n8k16_a_f16_1[2]), "r"(_mma_sync_m16n8k16_a_f16_1[3]), "r"(_mma_sync_m16n8k16_b_6[0]), "r"(_mma_sync_m16n8k16_b_6[1]));
                    }
                    int sf_linear_19 = 64 + base_n + 48;
                    uint8_t scale_byte_20 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_19 / 4 * 4));
                        scale_byte_20 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_19 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_21 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_14;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_21 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_20)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_14) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_15;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_21 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_20)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_15) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_7[2];
                    _mma_sync_m16n8k16_b_7[0] = _fp4_dequant_x2_14;
                    _mma_sync_m16n8k16_b_7[1] = _fp4_dequant_x2_15;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc3[0]), "+f"(acc3[1]), "+f"(acc3[2]), "+f"(acc3[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_1[0]), "r"(_mma_sync_m16n8k16_a_f16_1[1]), "r"(_mma_sync_m16n8k16_a_f16_1[2]), "r"(_mma_sync_m16n8k16_a_f16_1[3]), "r"(_mma_sync_m16n8k16_b_7[0]), "r"(_mma_sync_m16n8k16_b_7[1]));
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"((a_base + ((m_warp * 16 + lane % 16) * 128 + (32 + lane / 16 * 8) * 2 ^ ((m_warp * 16 + lane % 16) * 128 + (32 + lane / 16 * 8) * 2 >> 7 & 7) << 4)))
                        : "memory");
                    uint32_t _mma_sync_m16n8k16_a_f16_2[4];
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_2[0]) : "r"(a_frag[0]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_2[1]) : "r"(a_frag[1]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_2[2]) : "r"(a_frag[2]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_2[3]) : "r"(a_frag[3]));
                    int u32_pos_22 = n_warp * 64 + lane * 2;
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
                        : "r"(b_base + (256 + u32_pos_22) * 4));
                    int sf_linear_23 = 128 + base_n;
                    uint8_t scale_byte_24 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_23 / 4 * 4));
                        scale_byte_24 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_23 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_25 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_16;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_25 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_24)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_16) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_17;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_25 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_24)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_17) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_8[2];
                    _mma_sync_m16n8k16_b_8[0] = _fp4_dequant_x2_16;
                    _mma_sync_m16n8k16_b_8[1] = _fp4_dequant_x2_17;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_2[0]), "r"(_mma_sync_m16n8k16_a_f16_2[1]), "r"(_mma_sync_m16n8k16_a_f16_2[2]), "r"(_mma_sync_m16n8k16_a_f16_2[3]), "r"(_mma_sync_m16n8k16_b_8[0]), "r"(_mma_sync_m16n8k16_b_8[1]));
                    }
                    int sf_linear_26 = 128 + base_n + 16;
                    uint8_t scale_byte_27 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_26 / 4 * 4));
                        scale_byte_27 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_26 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_28 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_18;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_28 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_27)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_18) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_19;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_28 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_27)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_19) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_9[2];
                    _mma_sync_m16n8k16_b_9[0] = _fp4_dequant_x2_18;
                    _mma_sync_m16n8k16_b_9[1] = _fp4_dequant_x2_19;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_2[0]), "r"(_mma_sync_m16n8k16_a_f16_2[1]), "r"(_mma_sync_m16n8k16_a_f16_2[2]), "r"(_mma_sync_m16n8k16_a_f16_2[3]), "r"(_mma_sync_m16n8k16_b_9[0]), "r"(_mma_sync_m16n8k16_b_9[1]));
                    }
                    int sf_linear_29 = 128 + base_n + 32;
                    uint8_t scale_byte_30 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_29 / 4 * 4));
                        scale_byte_30 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_29 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_31 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_20;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_31 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_30)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_20) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_21;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_31 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_30)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_21) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_10[2];
                    _mma_sync_m16n8k16_b_10[0] = _fp4_dequant_x2_20;
                    _mma_sync_m16n8k16_b_10[1] = _fp4_dequant_x2_21;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc2[0]), "+f"(acc2[1]), "+f"(acc2[2]), "+f"(acc2[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_2[0]), "r"(_mma_sync_m16n8k16_a_f16_2[1]), "r"(_mma_sync_m16n8k16_a_f16_2[2]), "r"(_mma_sync_m16n8k16_a_f16_2[3]), "r"(_mma_sync_m16n8k16_b_10[0]), "r"(_mma_sync_m16n8k16_b_10[1]));
                    }
                    int sf_linear_32 = 128 + base_n + 48;
                    uint8_t scale_byte_33 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_32 / 4 * 4));
                        scale_byte_33 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_32 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_34 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_22;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_34 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_33)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_22) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_23;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_34 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_33)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_23) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_11[2];
                    _mma_sync_m16n8k16_b_11[0] = _fp4_dequant_x2_22;
                    _mma_sync_m16n8k16_b_11[1] = _fp4_dequant_x2_23;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc3[0]), "+f"(acc3[1]), "+f"(acc3[2]), "+f"(acc3[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_2[0]), "r"(_mma_sync_m16n8k16_a_f16_2[1]), "r"(_mma_sync_m16n8k16_a_f16_2[2]), "r"(_mma_sync_m16n8k16_a_f16_2[3]), "r"(_mma_sync_m16n8k16_b_11[0]), "r"(_mma_sync_m16n8k16_b_11[1]));
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"((a_base + ((m_warp * 16 + lane % 16) * 128 + (48 + lane / 16 * 8) * 2 ^ ((m_warp * 16 + lane % 16) * 128 + (48 + lane / 16 * 8) * 2 >> 7 & 7) << 4)))
                        : "memory");
                    uint32_t _mma_sync_m16n8k16_a_f16_3[4];
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_3[0]) : "r"(a_frag[0]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_3[1]) : "r"(a_frag[1]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_3[2]) : "r"(a_frag[2]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_3[3]) : "r"(a_frag[3]));
                    int u32_pos_35 = n_warp * 64 + lane * 2;
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
                        : "r"(b_base + (384 + u32_pos_35) * 4));
                    int sf_linear_36 = 192 + base_n;
                    uint8_t scale_byte_37 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_36 / 4 * 4));
                        scale_byte_37 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_36 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_38 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_24;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_38 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_37)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_24) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_25;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_38 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_37)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_25) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_12[2];
                    _mma_sync_m16n8k16_b_12[0] = _fp4_dequant_x2_24;
                    _mma_sync_m16n8k16_b_12[1] = _fp4_dequant_x2_25;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_3[0]), "r"(_mma_sync_m16n8k16_a_f16_3[1]), "r"(_mma_sync_m16n8k16_a_f16_3[2]), "r"(_mma_sync_m16n8k16_a_f16_3[3]), "r"(_mma_sync_m16n8k16_b_12[0]), "r"(_mma_sync_m16n8k16_b_12[1]));
                    }
                    int sf_linear_39 = 192 + base_n + 16;
                    uint8_t scale_byte_40 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_39 / 4 * 4));
                        scale_byte_40 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_39 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_41 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_26;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_41 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_40)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_26) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_27;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_41 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_40)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_27) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_13[2];
                    _mma_sync_m16n8k16_b_13[0] = _fp4_dequant_x2_26;
                    _mma_sync_m16n8k16_b_13[1] = _fp4_dequant_x2_27;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_3[0]), "r"(_mma_sync_m16n8k16_a_f16_3[1]), "r"(_mma_sync_m16n8k16_a_f16_3[2]), "r"(_mma_sync_m16n8k16_a_f16_3[3]), "r"(_mma_sync_m16n8k16_b_13[0]), "r"(_mma_sync_m16n8k16_b_13[1]));
                    }
                    int sf_linear_42 = 192 + base_n + 32;
                    uint8_t scale_byte_43 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_42 / 4 * 4));
                        scale_byte_43 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_42 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_44 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_28;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_44 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_43)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_28) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_29;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_44 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_43)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_29) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_14[2];
                    _mma_sync_m16n8k16_b_14[0] = _fp4_dequant_x2_28;
                    _mma_sync_m16n8k16_b_14[1] = _fp4_dequant_x2_29;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc2[0]), "+f"(acc2[1]), "+f"(acc2[2]), "+f"(acc2[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_3[0]), "r"(_mma_sync_m16n8k16_a_f16_3[1]), "r"(_mma_sync_m16n8k16_a_f16_3[2]), "r"(_mma_sync_m16n8k16_a_f16_3[3]), "r"(_mma_sync_m16n8k16_b_14[0]), "r"(_mma_sync_m16n8k16_b_14[1]));
                    }
                    int sf_linear_45 = 192 + base_n + 48;
                    uint8_t scale_byte_46 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_45 / 4 * 4));
                        scale_byte_46 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_45 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_47 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_30;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_47 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_46)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_30) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_31;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_47 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_46)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_31) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_15[2];
                    _mma_sync_m16n8k16_b_15[0] = _fp4_dequant_x2_30;
                    _mma_sync_m16n8k16_b_15[1] = _fp4_dequant_x2_31;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc3[0]), "+f"(acc3[1]), "+f"(acc3[2]), "+f"(acc3[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_3[0]), "r"(_mma_sync_m16n8k16_a_f16_3[1]), "r"(_mma_sync_m16n8k16_a_f16_3[2]), "r"(_mma_sync_m16n8k16_a_f16_3[3]), "r"(_mma_sync_m16n8k16_b_15[0]), "r"(_mma_sync_m16n8k16_b_15[1]));
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"((a_base + 2048 + ((m_warp * 16 + lane % 16) * 128 + lane / 16 * 8 * 2 ^ ((m_warp * 16 + lane % 16) * 128 + lane / 16 * 8 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    uint32_t _mma_sync_m16n8k16_a_f16_4[4];
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_4[0]) : "r"(a_frag[0]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_4[1]) : "r"(a_frag[1]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_4[2]) : "r"(a_frag[2]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_4[3]) : "r"(a_frag[3]));
                    int u32_pos_48 = n_warp * 64 + lane * 2;
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
                        : "r"(b_base + (512 + u32_pos_48) * 4));
                    int sf_linear_49 = 256 + base_n;
                    uint8_t scale_byte_50 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_49 / 4 * 4));
                        scale_byte_50 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_49 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_51 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_32;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_51 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_50)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_32) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_33;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_51 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_50)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_33) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_16[2];
                    _mma_sync_m16n8k16_b_16[0] = _fp4_dequant_x2_32;
                    _mma_sync_m16n8k16_b_16[1] = _fp4_dequant_x2_33;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_4[0]), "r"(_mma_sync_m16n8k16_a_f16_4[1]), "r"(_mma_sync_m16n8k16_a_f16_4[2]), "r"(_mma_sync_m16n8k16_a_f16_4[3]), "r"(_mma_sync_m16n8k16_b_16[0]), "r"(_mma_sync_m16n8k16_b_16[1]));
                    }
                    int sf_linear_52 = 256 + base_n + 16;
                    uint8_t scale_byte_53 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_52 / 4 * 4));
                        scale_byte_53 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_52 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_54 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_34;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_54 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_53)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_34) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_35;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_54 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_53)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_35) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_17[2];
                    _mma_sync_m16n8k16_b_17[0] = _fp4_dequant_x2_34;
                    _mma_sync_m16n8k16_b_17[1] = _fp4_dequant_x2_35;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_4[0]), "r"(_mma_sync_m16n8k16_a_f16_4[1]), "r"(_mma_sync_m16n8k16_a_f16_4[2]), "r"(_mma_sync_m16n8k16_a_f16_4[3]), "r"(_mma_sync_m16n8k16_b_17[0]), "r"(_mma_sync_m16n8k16_b_17[1]));
                    }
                    int sf_linear_55 = 256 + base_n + 32;
                    uint8_t scale_byte_56 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_55 / 4 * 4));
                        scale_byte_56 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_55 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_57 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_36;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_57 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_56)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_36) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_37;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_57 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_56)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_37) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_18[2];
                    _mma_sync_m16n8k16_b_18[0] = _fp4_dequant_x2_36;
                    _mma_sync_m16n8k16_b_18[1] = _fp4_dequant_x2_37;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc2[0]), "+f"(acc2[1]), "+f"(acc2[2]), "+f"(acc2[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_4[0]), "r"(_mma_sync_m16n8k16_a_f16_4[1]), "r"(_mma_sync_m16n8k16_a_f16_4[2]), "r"(_mma_sync_m16n8k16_a_f16_4[3]), "r"(_mma_sync_m16n8k16_b_18[0]), "r"(_mma_sync_m16n8k16_b_18[1]));
                    }
                    int sf_linear_58 = 256 + base_n + 48;
                    uint8_t scale_byte_59 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_58 / 4 * 4));
                        scale_byte_59 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_58 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_60 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_38;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_60 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_59)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_38) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_39;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_60 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_59)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_39) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_19[2];
                    _mma_sync_m16n8k16_b_19[0] = _fp4_dequant_x2_38;
                    _mma_sync_m16n8k16_b_19[1] = _fp4_dequant_x2_39;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc3[0]), "+f"(acc3[1]), "+f"(acc3[2]), "+f"(acc3[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_4[0]), "r"(_mma_sync_m16n8k16_a_f16_4[1]), "r"(_mma_sync_m16n8k16_a_f16_4[2]), "r"(_mma_sync_m16n8k16_a_f16_4[3]), "r"(_mma_sync_m16n8k16_b_19[0]), "r"(_mma_sync_m16n8k16_b_19[1]));
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"((a_base + 2048 + ((m_warp * 16 + lane % 16) * 128 + (16 + lane / 16 * 8) * 2 ^ ((m_warp * 16 + lane % 16) * 128 + (16 + lane / 16 * 8) * 2 >> 7 & 7) << 4)))
                        : "memory");
                    uint32_t _mma_sync_m16n8k16_a_f16_5[4];
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_5[0]) : "r"(a_frag[0]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_5[1]) : "r"(a_frag[1]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_5[2]) : "r"(a_frag[2]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_5[3]) : "r"(a_frag[3]));
                    int u32_pos_61 = n_warp * 64 + lane * 2;
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
                        : "r"(b_base + (640 + u32_pos_61) * 4));
                    int sf_linear_62 = 320 + base_n;
                    uint8_t scale_byte_63 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_62 / 4 * 4));
                        scale_byte_63 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_62 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_64 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_40;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_64 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_63)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_40) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_41;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_64 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_63)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_41) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_20[2];
                    _mma_sync_m16n8k16_b_20[0] = _fp4_dequant_x2_40;
                    _mma_sync_m16n8k16_b_20[1] = _fp4_dequant_x2_41;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_5[0]), "r"(_mma_sync_m16n8k16_a_f16_5[1]), "r"(_mma_sync_m16n8k16_a_f16_5[2]), "r"(_mma_sync_m16n8k16_a_f16_5[3]), "r"(_mma_sync_m16n8k16_b_20[0]), "r"(_mma_sync_m16n8k16_b_20[1]));
                    }
                    int sf_linear_65 = 320 + base_n + 16;
                    uint8_t scale_byte_66 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_65 / 4 * 4));
                        scale_byte_66 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_65 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_67 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_42;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_67 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_66)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_42) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_43;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_67 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_66)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_43) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_21[2];
                    _mma_sync_m16n8k16_b_21[0] = _fp4_dequant_x2_42;
                    _mma_sync_m16n8k16_b_21[1] = _fp4_dequant_x2_43;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_5[0]), "r"(_mma_sync_m16n8k16_a_f16_5[1]), "r"(_mma_sync_m16n8k16_a_f16_5[2]), "r"(_mma_sync_m16n8k16_a_f16_5[3]), "r"(_mma_sync_m16n8k16_b_21[0]), "r"(_mma_sync_m16n8k16_b_21[1]));
                    }
                    int sf_linear_68 = 320 + base_n + 32;
                    uint8_t scale_byte_69 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_68 / 4 * 4));
                        scale_byte_69 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_68 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_70 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_44;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_70 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_69)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_44) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_45;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_70 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_69)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_45) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_22[2];
                    _mma_sync_m16n8k16_b_22[0] = _fp4_dequant_x2_44;
                    _mma_sync_m16n8k16_b_22[1] = _fp4_dequant_x2_45;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc2[0]), "+f"(acc2[1]), "+f"(acc2[2]), "+f"(acc2[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_5[0]), "r"(_mma_sync_m16n8k16_a_f16_5[1]), "r"(_mma_sync_m16n8k16_a_f16_5[2]), "r"(_mma_sync_m16n8k16_a_f16_5[3]), "r"(_mma_sync_m16n8k16_b_22[0]), "r"(_mma_sync_m16n8k16_b_22[1]));
                    }
                    int sf_linear_71 = 320 + base_n + 48;
                    uint8_t scale_byte_72 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_71 / 4 * 4));
                        scale_byte_72 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_71 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_73 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_46;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_73 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_72)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_46) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_47;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_73 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_72)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_47) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_23[2];
                    _mma_sync_m16n8k16_b_23[0] = _fp4_dequant_x2_46;
                    _mma_sync_m16n8k16_b_23[1] = _fp4_dequant_x2_47;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc3[0]), "+f"(acc3[1]), "+f"(acc3[2]), "+f"(acc3[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_5[0]), "r"(_mma_sync_m16n8k16_a_f16_5[1]), "r"(_mma_sync_m16n8k16_a_f16_5[2]), "r"(_mma_sync_m16n8k16_a_f16_5[3]), "r"(_mma_sync_m16n8k16_b_23[0]), "r"(_mma_sync_m16n8k16_b_23[1]));
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"((a_base + 2048 + ((m_warp * 16 + lane % 16) * 128 + (32 + lane / 16 * 8) * 2 ^ ((m_warp * 16 + lane % 16) * 128 + (32 + lane / 16 * 8) * 2 >> 7 & 7) << 4)))
                        : "memory");
                    uint32_t _mma_sync_m16n8k16_a_f16_6[4];
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_6[0]) : "r"(a_frag[0]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_6[1]) : "r"(a_frag[1]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_6[2]) : "r"(a_frag[2]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_6[3]) : "r"(a_frag[3]));
                    int u32_pos_74 = n_warp * 64 + lane * 2;
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
                        : "r"(b_base + (768 + u32_pos_74) * 4));
                    int sf_linear_75 = 384 + base_n;
                    uint8_t scale_byte_76 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_75 / 4 * 4));
                        scale_byte_76 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_75 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_77 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_48;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_77 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_76)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_48) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_49;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_77 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_76)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_49) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_24[2];
                    _mma_sync_m16n8k16_b_24[0] = _fp4_dequant_x2_48;
                    _mma_sync_m16n8k16_b_24[1] = _fp4_dequant_x2_49;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_6[0]), "r"(_mma_sync_m16n8k16_a_f16_6[1]), "r"(_mma_sync_m16n8k16_a_f16_6[2]), "r"(_mma_sync_m16n8k16_a_f16_6[3]), "r"(_mma_sync_m16n8k16_b_24[0]), "r"(_mma_sync_m16n8k16_b_24[1]));
                    }
                    int sf_linear_78 = 384 + base_n + 16;
                    uint8_t scale_byte_79 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_78 / 4 * 4));
                        scale_byte_79 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_78 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_80 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_50;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_80 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_79)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_50) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_51;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_80 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_79)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_51) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_25[2];
                    _mma_sync_m16n8k16_b_25[0] = _fp4_dequant_x2_50;
                    _mma_sync_m16n8k16_b_25[1] = _fp4_dequant_x2_51;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_6[0]), "r"(_mma_sync_m16n8k16_a_f16_6[1]), "r"(_mma_sync_m16n8k16_a_f16_6[2]), "r"(_mma_sync_m16n8k16_a_f16_6[3]), "r"(_mma_sync_m16n8k16_b_25[0]), "r"(_mma_sync_m16n8k16_b_25[1]));
                    }
                    int sf_linear_81 = 384 + base_n + 32;
                    uint8_t scale_byte_82 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_81 / 4 * 4));
                        scale_byte_82 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_81 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_83 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_52;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_83 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_82)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_52) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_53;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_83 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_82)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_53) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_26[2];
                    _mma_sync_m16n8k16_b_26[0] = _fp4_dequant_x2_52;
                    _mma_sync_m16n8k16_b_26[1] = _fp4_dequant_x2_53;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc2[0]), "+f"(acc2[1]), "+f"(acc2[2]), "+f"(acc2[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_6[0]), "r"(_mma_sync_m16n8k16_a_f16_6[1]), "r"(_mma_sync_m16n8k16_a_f16_6[2]), "r"(_mma_sync_m16n8k16_a_f16_6[3]), "r"(_mma_sync_m16n8k16_b_26[0]), "r"(_mma_sync_m16n8k16_b_26[1]));
                    }
                    int sf_linear_84 = 384 + base_n + 48;
                    uint8_t scale_byte_85 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_84 / 4 * 4));
                        scale_byte_85 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_84 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_86 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_54;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_86 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_85)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_54) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_55;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_86 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_85)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_55) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_27[2];
                    _mma_sync_m16n8k16_b_27[0] = _fp4_dequant_x2_54;
                    _mma_sync_m16n8k16_b_27[1] = _fp4_dequant_x2_55;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc3[0]), "+f"(acc3[1]), "+f"(acc3[2]), "+f"(acc3[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_6[0]), "r"(_mma_sync_m16n8k16_a_f16_6[1]), "r"(_mma_sync_m16n8k16_a_f16_6[2]), "r"(_mma_sync_m16n8k16_a_f16_6[3]), "r"(_mma_sync_m16n8k16_b_27[0]), "r"(_mma_sync_m16n8k16_b_27[1]));
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                        : "r"((a_base + 2048 + ((m_warp * 16 + lane % 16) * 128 + (48 + lane / 16 * 8) * 2 ^ ((m_warp * 16 + lane % 16) * 128 + (48 + lane / 16 * 8) * 2 >> 7 & 7) << 4)))
                        : "memory");
                    uint32_t _mma_sync_m16n8k16_a_f16_7[4];
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_7[0]) : "r"(a_frag[0]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_7[1]) : "r"(a_frag[1]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_7[2]) : "r"(a_frag[2]));
                    asm(
                        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                        : "=r"(_mma_sync_m16n8k16_a_f16_7[3]) : "r"(a_frag[3]));
                    int u32_pos_87 = n_warp * 64 + lane * 2;
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1]))
                        : "r"(b_base + (896 + u32_pos_87) * 4));
                    int sf_linear_88 = 448 + base_n;
                    uint8_t scale_byte_89 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_88 / 4 * 4));
                        scale_byte_89 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_88 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_90 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_56;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_90 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_89)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_56) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_57;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_90 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_89)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_57) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_28[2];
                    _mma_sync_m16n8k16_b_28[0] = _fp4_dequant_x2_56;
                    _mma_sync_m16n8k16_b_28[1] = _fp4_dequant_x2_57;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_7[0]), "r"(_mma_sync_m16n8k16_a_f16_7[1]), "r"(_mma_sync_m16n8k16_a_f16_7[2]), "r"(_mma_sync_m16n8k16_a_f16_7[3]), "r"(_mma_sync_m16n8k16_b_28[0]), "r"(_mma_sync_m16n8k16_b_28[1]));
                    }
                    int sf_linear_91 = 448 + base_n + 16;
                    uint8_t scale_byte_92 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_91 / 4 * 4));
                        scale_byte_92 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_91 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_93 = ((1) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_58;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_93 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_92)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_58) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_59;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_93 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_92)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_59) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_29[2];
                    _mma_sync_m16n8k16_b_29[0] = _fp4_dequant_x2_58;
                    _mma_sync_m16n8k16_b_29[1] = _fp4_dequant_x2_59;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_7[0]), "r"(_mma_sync_m16n8k16_a_f16_7[1]), "r"(_mma_sync_m16n8k16_a_f16_7[2]), "r"(_mma_sync_m16n8k16_a_f16_7[3]), "r"(_mma_sync_m16n8k16_b_29[0]), "r"(_mma_sync_m16n8k16_b_29[1]));
                    }
                    int sf_linear_94 = 448 + base_n + 32;
                    uint8_t scale_byte_95 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_94 / 4 * 4));
                        scale_byte_95 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_94 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_96 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_60;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_96 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_95)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_60) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_61;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_96 >> 8 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_95)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_61) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_30[2];
                    _mma_sync_m16n8k16_b_30[0] = _fp4_dequant_x2_60;
                    _mma_sync_m16n8k16_b_30[1] = _fp4_dequant_x2_61;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc2[0]), "+f"(acc2[1]), "+f"(acc2[2]), "+f"(acc2[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_7[0]), "r"(_mma_sync_m16n8k16_a_f16_7[1]), "r"(_mma_sync_m16n8k16_a_f16_7[2]), "r"(_mma_sync_m16n8k16_a_f16_7[3]), "r"(_mma_sync_m16n8k16_b_30[0]), "r"(_mma_sync_m16n8k16_b_30[1]));
                    }
                    int sf_linear_97 = 448 + base_n + 48;
                    uint8_t scale_byte_98 = (uint8_t)0;
                    {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_97 / 4 * 4));
                        scale_byte_98 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_97 & 3) * 8) & 255);
                    }
                    unsigned int packed_word_99 = ((0) ? raw[0] : raw[1]);
                    uint32_t _fp4_dequant_x2_62;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_99 >> 16 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_98)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_62) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _fp4_dequant_x2_63;
                    {
                        uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(packed_word_99 >> 24 & 255)) & 0xFFu);
                        uint32_t _fp4_x16x2;
                        asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                            "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                            "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                            : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                        uint32_t _scale_byte = ((uint32_t)(scale_byte_98)) & 0xFFu;
                        uint32_t _scale_f16x2;
                        asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                        uint32_t _scale_x16x2 = _scale_f16x2;
                        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_63) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                    }
                    uint32_t _mma_sync_m16n8k16_b_31[2];
                    _mma_sync_m16n8k16_b_31[0] = _fp4_dequant_x2_62;
                    _mma_sync_m16n8k16_b_31[1] = _fp4_dequant_x2_63;
                    {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc3[0]), "+f"(acc3[1]), "+f"(acc3[2]), "+f"(acc3[3])
                            : "r"(_mma_sync_m16n8k16_a_f16_7[0]), "r"(_mma_sync_m16n8k16_a_f16_7[1]), "r"(_mma_sync_m16n8k16_a_f16_7[2]), "r"(_mma_sync_m16n8k16_a_f16_7[3]), "r"(_mma_sync_m16n8k16_b_31[0]), "r"(_mma_sync_m16n8k16_b_31[1]));
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(ab_free_addr + (compute_stage) * 8);
                    }
                    compute_stage += 1;
                    if (compute_stage == 16) { compute_stage = 0; _phase_ab_full ^= 1; }
                }
                float alpha_value = 1.0f;
                epi01[0] = acc0[0] * alpha_value;
                epi01[4] = acc1[0] * alpha_value;
                epi23[0] = acc2[0] * alpha_value;
                epi23[4] = acc3[0] * alpha_value;
                epi01[1] = acc0[1] * alpha_value;
                epi01[5] = acc1[1] * alpha_value;
                epi23[1] = acc2[1] * alpha_value;
                epi23[5] = acc3[1] * alpha_value;
                epi01[2] = acc0[2] * alpha_value;
                epi01[6] = acc1[2] * alpha_value;
                epi23[2] = acc2[2] * alpha_value;
                epi23[6] = acc3[2] * alpha_value;
                epi01[3] = acc0[3] * alpha_value;
                epi01[7] = acc1[3] * alpha_value;
                epi23[3] = acc2[3] * alpha_value;
                epi23[7] = acc3[3] * alpha_value;
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(epi01[_lp*2 + 0], epi01[_lp*2+1 + 0]));
                    packed01[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(epi23[_lp*2 + 0], epi23[_lp*2+1 + 0]));
                    packed23[_lp] = *(uint32_t*)&_bf2;
                }
                int lane_row = lane % 16;
                int lane_col = lane / 16 * 16;
                int n_stripe = warp_id_in_role * 8;
                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(warp_mma_c_addr + (unsigned int)((m_warp * 16 + lane_row) * 128 + (n_stripe + lane_col) * 2 ^ ((m_warp * 16 + lane_row) * 128 + (n_stripe + lane_col) * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&packed01[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed01[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed01[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed01[3]))
                    : "memory");
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)(warp_mma_c_addr + (unsigned int)((m_warp * 16 + lane_row) * 128 + (n_stripe + 32 + lane_col) * 2 ^ ((m_warp * 16 + lane_row) * 128 + (n_stripe + 32 + lane_col) * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&packed23[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed23[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed23[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed23[3]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 64;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        tma_store_2d((&C), off_n, off_m, warp_mma_c_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
                asm volatile("barrier.sync 8, 64;" ::: "memory");
            }
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                }
            }
        }
    }
    // ---- Role: dma ----
    if (warp == 2) {
        { // dma_main
            int grid_n_1 = (N + 64 - 1) / 64;
            int grid_m_1 = (M + 16 - 1) / 16;
            int total_tiles_1 = grid_m_1 * grid_n_1;
            unsigned int load_stage = 0;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (unsigned int work_1 = blockIdx.x; work_1 < total_tiles_1; work_1 += gridDim.x) {
                int tile_m_1 = work_1 / (unsigned int)grid_n_1;
                int tile_n_1 = work_1 - (unsigned int)(tile_m_1 * grid_n_1);
                int off_m_1 = tile_m_1 * 16;
                int off_n_1 = tile_n_1 * 64;
                int k_tiles_1 = (K + 128 - 1) / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int kt_1 = 0; kt_1 < k_tiles_1; kt_1++) {
                        mbarrier_wait(ab_free_addr + (load_stage) * 8, _phase_ab_free);
                        tma_3d_gmem2smem(warp_mma_a_addr + load_stage * 9216, (&A), 0, off_m_1, kt_1 * 2, ab_full_addr + (load_stage) * 8);
                        tma_2d_gmem2smem(warp_mma_b_packed_addr + load_stage * 9216, (&B), off_n_1 * 2, kt_1 * 8, ab_full_addr + (load_stage) * 8);
                        {
                            tma_2d_gmem2smem(warp_mma_b_scale_addr + load_stage * 9216, (&B_descale), off_n_1, kt_1 * 8, ab_full_addr + (load_stage) * 8);
                        }
                        mbarrier_arrive_expect_tx(ab_full_addr + (load_stage) * 8, 8192 + ((1) ? 512 : 0));
                        load_stage += 1;
                        if (load_stage == 16) { load_stage = 0; _phase_ab_free ^= 1; }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
