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
#define NUM_MAIN_STAGES 1
#define SMEM_V41_CUTE_K256_P2_A_OFF 0
#define SMEM_V41_CUTE_K256_P2_A_STAGE_BYTES 8192
#define SMEM_V41_CUTE_K256_P2_A_STRIDE 17408
#define SMEM_V41_CUTE_K256_P2_B_OFF 8192
#define SMEM_V41_CUTE_K256_P2_B_STAGE_BYTES 8192
#define SMEM_V41_CUTE_K256_P2_B_STRIDE 17408
#define SMEM_V41_CUTE_K256_P2_SCALE_OFF 16384
#define SMEM_V41_CUTE_K256_P2_SCALE_STAGE_BYTES 1024
#define SMEM_V41_CUTE_K256_P2_SCALE_STRIDE 17408
#define SMEM_TOTAL 34816
#define THREADS 128
#define HAS_ALPHA 1
#define ENABLE_PDL 1
#define EXACT_ROW7 0

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


__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_blackwell_bf16_fp4_5d47a298f721f0fb3b82(__nv_bfloat16* __restrict__ A, int* __restrict__ B, uint8_t* __restrict__ B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* v41_cute_k256_p2_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int v41_cute_k256_p2_a_addr = smem + 0;
    int* v41_cute_k256_p2_b = reinterpret_cast<int*>(smem_raw + 8192);
    const int v41_cute_k256_p2_b_addr = smem + 8192;
    uint8_t* v41_cute_k256_p2_scale = reinterpret_cast<uint8_t*>(smem_raw + 16384);
    const int v41_cute_k256_p2_scale_addr = smem + 16384;

    // === Task calls (dependency order) ===
    int grid_n = (N + 64 - 1) / 64;
    int grid_m = (M + 16 - 1) / 16;
    int total_tiles = grid_m * grid_n;
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    #pragma unroll 1
    for (unsigned int work = blockIdx.x; work < total_tiles; work += gridDim.x) {
        int tile_m = work / (unsigned int)grid_n;
        int tile_n = work - (unsigned int)(tile_m * grid_n);
        int off_m = tile_m * 16;
        int off_n = tile_n * 64;
        int packed_off_n = tile_n * 64;
        int k_tiles = (K + 256 - 1) / 256;
        int k_tiles_0 = (K + 256 - 1) / 256;
        bool stage_valid = k_tiles_0 > 0;
        int _min_0 = ((0) < (k_tiles_0 - 1) ? (0) : (k_tiles_0 - 1));
        int safe_local_kt = _min_0;
        int total_k_groups = K / 16;
        int chunk = tid;
        int local_m = chunk / 32;
        int local_k = chunk % 32 * 8;
        int global_m = off_m + local_m;
        int global_k = safe_local_kt * 256 + local_k;
        int _min_1 = ((global_m) < (M - 1) ? (global_m) : (M - 1));
        int safe_m = _min_1;
        int _min_2 = ((global_k) < (K - 8) ? (global_k) : (K - 8));
        int safe_k = _min_2;
        bool valid_a = stage_valid && global_m < M && global_k < K;
        int a_plane = local_k / 64;
        int a_plane_k = local_k - a_plane * 64;
        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_a_addr + (unsigned int)(a_plane * 16 * 64 * 2) + (unsigned int)(local_m * 128 + a_plane_k * 2 ^ (local_m * 128 + a_plane_k * 2 >> 7 & 7) << 4))), "l"(A + (safe_m * K + safe_k)), "r"((valid_a) ? 16 : 0));
        int chunk_1 = tid + 128;
        int local_m_2 = chunk_1 / 32;
        int local_k_3 = chunk_1 % 32 * 8;
        int global_m_4 = off_m + local_m_2;
        int global_k_5 = safe_local_kt * 256 + local_k_3;
        int _min_3 = ((global_m_4) < (M - 1) ? (global_m_4) : (M - 1));
        int safe_m_6 = _min_3;
        int _min_4 = ((global_k_5) < (K - 8) ? (global_k_5) : (K - 8));
        int safe_k_7 = _min_4;
        bool valid_a_8 = stage_valid && global_m_4 < M && global_k_5 < K;
        int a_plane_9 = local_k_3 / 64;
        int a_plane_k_10 = local_k_3 - a_plane_9 * 64;
        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_a_addr + (unsigned int)(a_plane_9 * 16 * 64 * 2) + (unsigned int)(local_m_2 * 128 + a_plane_k_10 * 2 ^ (local_m_2 * 128 + a_plane_k_10 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_6 * K + safe_k_7)), "r"((valid_a_8) ? 16 : 0));
        int chunk_11 = tid + 256;
        int local_m_12 = chunk_11 / 32;
        int local_k_13 = chunk_11 % 32 * 8;
        int global_m_14 = off_m + local_m_12;
        int global_k_15 = safe_local_kt * 256 + local_k_13;
        int _min_5 = ((global_m_14) < (M - 1) ? (global_m_14) : (M - 1));
        int safe_m_16 = _min_5;
        int _min_6 = ((global_k_15) < (K - 8) ? (global_k_15) : (K - 8));
        int safe_k_17 = _min_6;
        bool valid_a_18 = stage_valid && global_m_14 < M && global_k_15 < K;
        int a_plane_19 = local_k_13 / 64;
        int a_plane_k_20 = local_k_13 - a_plane_19 * 64;
        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_a_addr + (unsigned int)(a_plane_19 * 16 * 64 * 2) + (unsigned int)(local_m_12 * 128 + a_plane_k_20 * 2 ^ (local_m_12 * 128 + a_plane_k_20 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_16 * K + safe_k_17)), "r"((valid_a_18) ? 16 : 0));
        int chunk_21 = tid + 384;
        int local_m_22 = chunk_21 / 32;
        int local_k_23 = chunk_21 % 32 * 8;
        int global_m_24 = off_m + local_m_22;
        int global_k_25 = safe_local_kt * 256 + local_k_23;
        int _min_7 = ((global_m_24) < (M - 1) ? (global_m_24) : (M - 1));
        int safe_m_26 = _min_7;
        int _min_8 = ((global_k_25) < (K - 8) ? (global_k_25) : (K - 8));
        int safe_k_27 = _min_8;
        bool valid_a_28 = stage_valid && global_m_24 < M && global_k_25 < K;
        int a_plane_29 = local_k_23 / 64;
        int a_plane_k_30 = local_k_23 - a_plane_29 * 64;
        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_a_addr + (unsigned int)(a_plane_29 * 16 * 64 * 2) + (unsigned int)(local_m_22 * 128 + a_plane_k_30 * 2 ^ (local_m_22 * 128 + a_plane_k_30 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_26 * K + safe_k_27)), "r"((valid_a_28) ? 16 : 0));
        int chunk_31 = tid;
        int local_k_group = chunk_31 / 32;
        int local_word = chunk_31 % 32 * 4;
        int packed_panel = local_word / 32;
        int packed_panel_word = local_word - packed_panel * 32;
        int packed_row = local_k_group * 4 + packed_panel;
        int global_k_group = safe_local_kt * 16 + local_k_group;
        int _min_9 = ((global_k_group) < (total_k_groups - 1) ? (global_k_group) : (total_k_groups - 1));
        int safe_k_group = _min_9;
        bool valid_b = stage_valid && global_k_group < total_k_groups;
        asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_b_addr + (unsigned int)(packed_row * 128 + packed_panel_word * 4 ^ (packed_row * 128 + packed_panel_word * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group * (N * 2) + packed_off_n * 2 + local_word)), "r"((valid_b) ? 16 : 0));
        int chunk_32 = tid + 128;
        int local_k_group_33 = chunk_32 / 32;
        int local_word_34 = chunk_32 % 32 * 4;
        int packed_panel_35 = local_word_34 / 32;
        int packed_panel_word_36 = local_word_34 - packed_panel_35 * 32;
        int packed_row_37 = local_k_group_33 * 4 + packed_panel_35;
        int global_k_group_38 = safe_local_kt * 16 + local_k_group_33;
        int _min_10 = ((global_k_group_38) < (total_k_groups - 1) ? (global_k_group_38) : (total_k_groups - 1));
        int safe_k_group_39 = _min_10;
        bool valid_b_40 = stage_valid && global_k_group_38 < total_k_groups;
        asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_b_addr + (unsigned int)(packed_row_37 * 128 + packed_panel_word_36 * 4 ^ (packed_row_37 * 128 + packed_panel_word_36 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_39 * (N * 2) + packed_off_n * 2 + local_word_34)), "r"((valid_b_40) ? 16 : 0));
        int chunk_41 = tid + 256;
        int local_k_group_42 = chunk_41 / 32;
        int local_word_43 = chunk_41 % 32 * 4;
        int packed_panel_44 = local_word_43 / 32;
        int packed_panel_word_45 = local_word_43 - packed_panel_44 * 32;
        int packed_row_46 = local_k_group_42 * 4 + packed_panel_44;
        int global_k_group_47 = safe_local_kt * 16 + local_k_group_42;
        int _min_11 = ((global_k_group_47) < (total_k_groups - 1) ? (global_k_group_47) : (total_k_groups - 1));
        int safe_k_group_48 = _min_11;
        bool valid_b_49 = stage_valid && global_k_group_47 < total_k_groups;
        asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_b_addr + (unsigned int)(packed_row_46 * 128 + packed_panel_word_45 * 4 ^ (packed_row_46 * 128 + packed_panel_word_45 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_48 * (N * 2) + packed_off_n * 2 + local_word_43)), "r"((valid_b_49) ? 16 : 0));
        int chunk_50 = tid + 384;
        int local_k_group_51 = chunk_50 / 32;
        int local_word_52 = chunk_50 % 32 * 4;
        int packed_panel_53 = local_word_52 / 32;
        int packed_panel_word_54 = local_word_52 - packed_panel_53 * 32;
        int packed_row_55 = local_k_group_51 * 4 + packed_panel_53;
        int global_k_group_56 = safe_local_kt * 16 + local_k_group_51;
        int _min_12 = ((global_k_group_56) < (total_k_groups - 1) ? (global_k_group_56) : (total_k_groups - 1));
        int safe_k_group_57 = _min_12;
        bool valid_b_58 = stage_valid && global_k_group_56 < total_k_groups;
        asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_b_addr + (unsigned int)(packed_row_55 * 128 + packed_panel_word_54 * 4 ^ (packed_row_55 * 128 + packed_panel_word_54 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_57 * (N * 2) + packed_off_n * 2 + local_word_52)), "r"((valid_b_58) ? 16 : 0));
        int _min_13 = (((int)lane) < (15) ? ((int)lane) : (15));
        int scale_chunk = warp * 16 + _min_13;
        int local_scale_k_group = scale_chunk / 4;
        int local_scale_n = scale_chunk % 4 * 16;
        int global_scale_k_group = safe_local_kt * 16 + local_scale_k_group;
        int _min_14 = ((global_scale_k_group) < (total_k_groups - 1) ? (global_scale_k_group) : (total_k_groups - 1));
        int safe_scale_k_group = _min_14;
        bool valid_scale = stage_valid && global_scale_k_group < total_k_groups;
        if (lane < 16) {
            asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                :: "r"(v41_cute_k256_p2_scale_addr + (unsigned int)(scale_chunk * 16)), "l"(B_descale + (safe_scale_k_group * N + off_n + local_scale_n)), "r"((valid_scale) ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;");
        int k_tiles_59 = (K + 256 - 1) / 256;
        bool stage_valid_60 = k_tiles_59 > 1;
        int _min_15 = ((1) < (k_tiles_59 - 1) ? (1) : (k_tiles_59 - 1));
        int safe_local_kt_61 = _min_15;
        int total_k_groups_62 = K / 16;
        int chunk_63 = tid;
        int local_m_64 = chunk_63 / 32;
        int local_k_65 = chunk_63 % 32 * 8;
        int global_m_66 = off_m + local_m_64;
        int global_k_67 = safe_local_kt_61 * 256 + local_k_65;
        int _min_16 = ((global_m_66) < (M - 1) ? (global_m_66) : (M - 1));
        int safe_m_68 = _min_16;
        int _min_17 = ((global_k_67) < (K - 8) ? (global_k_67) : (K - 8));
        int safe_k_69 = _min_17;
        bool valid_a_70 = stage_valid_60 && global_m_66 < M && global_k_67 < K;
        int a_plane_71 = local_k_65 / 64;
        int a_plane_k_72 = local_k_65 - a_plane_71 * 64;
        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_a_addr + 17408 + (unsigned int)(a_plane_71 * 16 * 64 * 2) + (unsigned int)(local_m_64 * 128 + a_plane_k_72 * 2 ^ (local_m_64 * 128 + a_plane_k_72 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_68 * K + safe_k_69)), "r"((valid_a_70) ? 16 : 0));
        int chunk_73 = tid + 128;
        int local_m_74 = chunk_73 / 32;
        int local_k_75 = chunk_73 % 32 * 8;
        int global_m_76 = off_m + local_m_74;
        int global_k_77 = safe_local_kt_61 * 256 + local_k_75;
        int _min_18 = ((global_m_76) < (M - 1) ? (global_m_76) : (M - 1));
        int safe_m_78 = _min_18;
        int _min_19 = ((global_k_77) < (K - 8) ? (global_k_77) : (K - 8));
        int safe_k_79 = _min_19;
        bool valid_a_80 = stage_valid_60 && global_m_76 < M && global_k_77 < K;
        int a_plane_81 = local_k_75 / 64;
        int a_plane_k_82 = local_k_75 - a_plane_81 * 64;
        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_a_addr + 17408 + (unsigned int)(a_plane_81 * 16 * 64 * 2) + (unsigned int)(local_m_74 * 128 + a_plane_k_82 * 2 ^ (local_m_74 * 128 + a_plane_k_82 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_78 * K + safe_k_79)), "r"((valid_a_80) ? 16 : 0));
        int chunk_83 = tid + 256;
        int local_m_84 = chunk_83 / 32;
        int local_k_85 = chunk_83 % 32 * 8;
        int global_m_86 = off_m + local_m_84;
        int global_k_87 = safe_local_kt_61 * 256 + local_k_85;
        int _min_20 = ((global_m_86) < (M - 1) ? (global_m_86) : (M - 1));
        int safe_m_88 = _min_20;
        int _min_21 = ((global_k_87) < (K - 8) ? (global_k_87) : (K - 8));
        int safe_k_89 = _min_21;
        bool valid_a_90 = stage_valid_60 && global_m_86 < M && global_k_87 < K;
        int a_plane_91 = local_k_85 / 64;
        int a_plane_k_92 = local_k_85 - a_plane_91 * 64;
        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_a_addr + 17408 + (unsigned int)(a_plane_91 * 16 * 64 * 2) + (unsigned int)(local_m_84 * 128 + a_plane_k_92 * 2 ^ (local_m_84 * 128 + a_plane_k_92 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_88 * K + safe_k_89)), "r"((valid_a_90) ? 16 : 0));
        int chunk_93 = tid + 384;
        int local_m_94 = chunk_93 / 32;
        int local_k_95 = chunk_93 % 32 * 8;
        int global_m_96 = off_m + local_m_94;
        int global_k_97 = safe_local_kt_61 * 256 + local_k_95;
        int _min_22 = ((global_m_96) < (M - 1) ? (global_m_96) : (M - 1));
        int safe_m_98 = _min_22;
        int _min_23 = ((global_k_97) < (K - 8) ? (global_k_97) : (K - 8));
        int safe_k_99 = _min_23;
        bool valid_a_100 = stage_valid_60 && global_m_96 < M && global_k_97 < K;
        int a_plane_101 = local_k_95 / 64;
        int a_plane_k_102 = local_k_95 - a_plane_101 * 64;
        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_a_addr + 17408 + (unsigned int)(a_plane_101 * 16 * 64 * 2) + (unsigned int)(local_m_94 * 128 + a_plane_k_102 * 2 ^ (local_m_94 * 128 + a_plane_k_102 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_98 * K + safe_k_99)), "r"((valid_a_100) ? 16 : 0));
        int chunk_103 = tid;
        int local_k_group_104 = chunk_103 / 32;
        int local_word_105 = chunk_103 % 32 * 4;
        int packed_panel_106 = local_word_105 / 32;
        int packed_panel_word_107 = local_word_105 - packed_panel_106 * 32;
        int packed_row_108 = local_k_group_104 * 4 + packed_panel_106;
        int global_k_group_109 = safe_local_kt_61 * 16 + local_k_group_104;
        int _min_24 = ((global_k_group_109) < (total_k_groups_62 - 1) ? (global_k_group_109) : (total_k_groups_62 - 1));
        int safe_k_group_110 = _min_24;
        bool valid_b_111 = stage_valid_60 && global_k_group_109 < total_k_groups_62;
        asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_b_addr + 17408 + (unsigned int)(packed_row_108 * 128 + packed_panel_word_107 * 4 ^ (packed_row_108 * 128 + packed_panel_word_107 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_110 * (N * 2) + packed_off_n * 2 + local_word_105)), "r"((valid_b_111) ? 16 : 0));
        int chunk_112 = tid + 128;
        int local_k_group_113 = chunk_112 / 32;
        int local_word_114 = chunk_112 % 32 * 4;
        int packed_panel_115 = local_word_114 / 32;
        int packed_panel_word_116 = local_word_114 - packed_panel_115 * 32;
        int packed_row_117 = local_k_group_113 * 4 + packed_panel_115;
        int global_k_group_118 = safe_local_kt_61 * 16 + local_k_group_113;
        int _min_25 = ((global_k_group_118) < (total_k_groups_62 - 1) ? (global_k_group_118) : (total_k_groups_62 - 1));
        int safe_k_group_119 = _min_25;
        bool valid_b_120 = stage_valid_60 && global_k_group_118 < total_k_groups_62;
        asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_b_addr + 17408 + (unsigned int)(packed_row_117 * 128 + packed_panel_word_116 * 4 ^ (packed_row_117 * 128 + packed_panel_word_116 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_119 * (N * 2) + packed_off_n * 2 + local_word_114)), "r"((valid_b_120) ? 16 : 0));
        int chunk_121 = tid + 256;
        int local_k_group_122 = chunk_121 / 32;
        int local_word_123 = chunk_121 % 32 * 4;
        int packed_panel_124 = local_word_123 / 32;
        int packed_panel_word_125 = local_word_123 - packed_panel_124 * 32;
        int packed_row_126 = local_k_group_122 * 4 + packed_panel_124;
        int global_k_group_127 = safe_local_kt_61 * 16 + local_k_group_122;
        int _min_26 = ((global_k_group_127) < (total_k_groups_62 - 1) ? (global_k_group_127) : (total_k_groups_62 - 1));
        int safe_k_group_128 = _min_26;
        bool valid_b_129 = stage_valid_60 && global_k_group_127 < total_k_groups_62;
        asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_b_addr + 17408 + (unsigned int)(packed_row_126 * 128 + packed_panel_word_125 * 4 ^ (packed_row_126 * 128 + packed_panel_word_125 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_128 * (N * 2) + packed_off_n * 2 + local_word_123)), "r"((valid_b_129) ? 16 : 0));
        int chunk_130 = tid + 384;
        int local_k_group_131 = chunk_130 / 32;
        int local_word_132 = chunk_130 % 32 * 4;
        int packed_panel_133 = local_word_132 / 32;
        int packed_panel_word_134 = local_word_132 - packed_panel_133 * 32;
        int packed_row_135 = local_k_group_131 * 4 + packed_panel_133;
        int global_k_group_136 = safe_local_kt_61 * 16 + local_k_group_131;
        int _min_27 = ((global_k_group_136) < (total_k_groups_62 - 1) ? (global_k_group_136) : (total_k_groups_62 - 1));
        int safe_k_group_137 = _min_27;
        bool valid_b_138 = stage_valid_60 && global_k_group_136 < total_k_groups_62;
        asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
            :: "r"((v41_cute_k256_p2_b_addr + 17408 + (unsigned int)(packed_row_135 * 128 + packed_panel_word_134 * 4 ^ (packed_row_135 * 128 + packed_panel_word_134 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_137 * (N * 2) + packed_off_n * 2 + local_word_132)), "r"((valid_b_138) ? 16 : 0));
        int _min_28 = (((int)lane) < (15) ? ((int)lane) : (15));
        int scale_chunk_139 = warp * 16 + _min_28;
        int local_scale_k_group_140 = scale_chunk_139 / 4;
        int local_scale_n_141 = scale_chunk_139 % 4 * 16;
        int global_scale_k_group_142 = safe_local_kt_61 * 16 + local_scale_k_group_140;
        int _min_29 = ((global_scale_k_group_142) < (total_k_groups_62 - 1) ? (global_scale_k_group_142) : (total_k_groups_62 - 1));
        int safe_scale_k_group_143 = _min_29;
        bool valid_scale_144 = stage_valid_60 && global_scale_k_group_142 < total_k_groups_62;
        if (lane < 16) {
            asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                :: "r"(v41_cute_k256_p2_scale_addr + 17408 + (unsigned int)(scale_chunk_139 * 16)), "l"(B_descale + (safe_scale_k_group_143 * N + off_n + local_scale_n_141)), "r"((valid_scale_144) ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;");
        unsigned int a_frag[4];
        unsigned int raw0[1];
        unsigned int raw1[1];
        unsigned int scale_word0[1];
        unsigned int scale_word1[1];
        unsigned int exact_raw0_ping[1];
        unsigned int exact_raw1_ping[1];
        unsigned int exact_scale0_ping[1];
        unsigned int exact_scale1_ping[1];
        unsigned int exact_raw0_pong[1];
        unsigned int exact_raw1_pong[1];
        unsigned int exact_scale0_pong[1];
        unsigned int exact_scale1_pong[1];
        float acc0[4];
        float acc1[4];
        acc0[0] = 0.0f;
        acc0[1] = 0.0f;
        acc0[2] = 0.0f;
        acc0[3] = 0.0f;
        acc1[0] = 0.0f;
        acc1[1] = 0.0f;
        acc1[2] = 0.0f;
        acc1[3] = 0.0f;
        #pragma unroll 1
        for (int local_kt = 0; local_kt < k_tiles; local_kt++) {
            int stage = local_kt % 2;
            asm volatile("cp.async.wait_group 1;");
            __syncthreads();
            int a_base = v41_cute_k256_p2_a_addr + (unsigned int)(stage * 17408);
            int b_base = v41_cute_k256_p2_b_addr + (unsigned int)(stage * 17408);
            int sf_base = v41_cute_k256_p2_scale_addr + (unsigned int)(stage * 17408);
            int tc_col = lane / 4;
            int base_n = tc_col;
            {
                int owner_half = warp / 2;
                int n_region = warp & 1;
                int u32_pos0 = lane * 2 + (unsigned int)owner_half;
                int packed_panel0 = u32_pos0 / 32;
                int packed_panel_word0 = u32_pos0 - packed_panel0 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + (packed_panel0 * 128 + packed_panel_word0 * 4 ^ (packed_panel0 * 128 + packed_panel_word0 * 4 >> 7 & 7) << 4))));
                int sf_linear0 = base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0 / 4 * 4));
                int u32_pos1 = 64 + lane * 2 + (unsigned int)owner_half;
                int packed_panel1 = u32_pos1 / 32;
                int packed_panel_word1 = u32_pos1 - packed_panel1 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + (packed_panel1 * 128 + packed_panel_word1 * 4 ^ (packed_panel1 * 128 + packed_panel_word1 * 4 >> 7 & 7) << 4))));
                int sf_linear1 = sf_linear0 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1 / 4 * 4));
                int a_group_base = a_base;
                int a_k_byte = lane / 16 * 8 * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base + (lane % 16 * 128 + (unsigned int)a_k_byte ^ (lane % 16 * 128 + (unsigned int)a_k_byte >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_16[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_16[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_16[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_16[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_16[3]) : "r"(a_frag[3]));
                int byte_shift = n_region * 16;
                uint8_t scale_byte0 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_64;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_64) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_65;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_65) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_32[2];
                _mma_sync_m16n8k16_b_32[0] = _fp4_dequant_x2_64;
                _mma_sync_m16n8k16_b_32[1] = _fp4_dequant_x2_65;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_16[0]), "r"(_mma_sync_m16n8k16_a_f16_16[1]), "r"(_mma_sync_m16n8k16_a_f16_16[2]), "r"(_mma_sync_m16n8k16_a_f16_16[3]), "r"(_mma_sync_m16n8k16_b_32[0]), "r"(_mma_sync_m16n8k16_b_32[1]));
                uint8_t scale_byte1 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_66;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_66) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_67;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_67) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_33[2];
                _mma_sync_m16n8k16_b_33[0] = _fp4_dequant_x2_66;
                _mma_sync_m16n8k16_b_33[1] = _fp4_dequant_x2_67;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_16[0]), "r"(_mma_sync_m16n8k16_a_f16_16[1]), "r"(_mma_sync_m16n8k16_a_f16_16[2]), "r"(_mma_sync_m16n8k16_a_f16_16[3]), "r"(_mma_sync_m16n8k16_b_33[0]), "r"(_mma_sync_m16n8k16_b_33[1]));
                int owner_half_0 = warp / 2;
                int n_region_1 = warp & 1;
                int u32_pos0_2 = lane * 2 + (unsigned int)owner_half_0;
                int packed_panel0_3 = u32_pos0_2 / 32;
                int packed_panel_word0_4 = u32_pos0_2 - packed_panel0_3 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((4 + packed_panel0_3) * 128 + packed_panel_word0_4 * 4 ^ ((4 + packed_panel0_3) * 128 + packed_panel_word0_4 * 4 >> 7 & 7) << 4))));
                int sf_linear0_5 = 64 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_5 / 4 * 4));
                int u32_pos1_6 = 64 + lane * 2 + (unsigned int)owner_half_0;
                int packed_panel1_7 = u32_pos1_6 / 32;
                int packed_panel_word1_8 = u32_pos1_6 - packed_panel1_7 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((4 + packed_panel1_7) * 128 + packed_panel_word1_8 * 4 ^ ((4 + packed_panel1_7) * 128 + packed_panel_word1_8 * 4 >> 7 & 7) << 4))));
                int sf_linear1_9 = sf_linear0_5 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_9 / 4 * 4));
                int a_group_base_10 = a_base;
                int a_k_byte_11 = (16 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_10 + (lane % 16 * 128 + (unsigned int)a_k_byte_11 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_11 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_17[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_17[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_17[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_17[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_17[3]) : "r"(a_frag[3]));
                int byte_shift_12 = n_region_1 * 16;
                uint8_t scale_byte0_13 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_5 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_68;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_12 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_13)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_68) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_69;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_12 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_13)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_69) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_34[2];
                _mma_sync_m16n8k16_b_34[0] = _fp4_dequant_x2_68;
                _mma_sync_m16n8k16_b_34[1] = _fp4_dequant_x2_69;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_17[0]), "r"(_mma_sync_m16n8k16_a_f16_17[1]), "r"(_mma_sync_m16n8k16_a_f16_17[2]), "r"(_mma_sync_m16n8k16_a_f16_17[3]), "r"(_mma_sync_m16n8k16_b_34[0]), "r"(_mma_sync_m16n8k16_b_34[1]));
                uint8_t scale_byte1_14 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_9 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_70;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_12 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_14)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_70) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_71;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_12 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_14)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_71) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_35[2];
                _mma_sync_m16n8k16_b_35[0] = _fp4_dequant_x2_70;
                _mma_sync_m16n8k16_b_35[1] = _fp4_dequant_x2_71;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_17[0]), "r"(_mma_sync_m16n8k16_a_f16_17[1]), "r"(_mma_sync_m16n8k16_a_f16_17[2]), "r"(_mma_sync_m16n8k16_a_f16_17[3]), "r"(_mma_sync_m16n8k16_b_35[0]), "r"(_mma_sync_m16n8k16_b_35[1]));
                int owner_half_15 = warp / 2;
                int n_region_16 = warp & 1;
                int u32_pos0_17 = lane * 2 + (unsigned int)owner_half_15;
                int packed_panel0_18 = u32_pos0_17 / 32;
                int packed_panel_word0_19 = u32_pos0_17 - packed_panel0_18 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((8 + packed_panel0_18) * 128 + packed_panel_word0_19 * 4 ^ ((8 + packed_panel0_18) * 128 + packed_panel_word0_19 * 4 >> 7 & 7) << 4))));
                int sf_linear0_20 = 128 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_20 / 4 * 4));
                int u32_pos1_21 = 64 + lane * 2 + (unsigned int)owner_half_15;
                int packed_panel1_22 = u32_pos1_21 / 32;
                int packed_panel_word1_23 = u32_pos1_21 - packed_panel1_22 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((8 + packed_panel1_22) * 128 + packed_panel_word1_23 * 4 ^ ((8 + packed_panel1_22) * 128 + packed_panel_word1_23 * 4 >> 7 & 7) << 4))));
                int sf_linear1_24 = sf_linear0_20 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_24 / 4 * 4));
                int a_group_base_25 = a_base;
                int a_k_byte_26 = (32 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_25 + (lane % 16 * 128 + (unsigned int)a_k_byte_26 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_26 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_18[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_18[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_18[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_18[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_18[3]) : "r"(a_frag[3]));
                int byte_shift_27 = n_region_16 * 16;
                uint8_t scale_byte0_28 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_20 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_72;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_27 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_28)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_72) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_73;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_27 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_28)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_73) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_36[2];
                _mma_sync_m16n8k16_b_36[0] = _fp4_dequant_x2_72;
                _mma_sync_m16n8k16_b_36[1] = _fp4_dequant_x2_73;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_18[0]), "r"(_mma_sync_m16n8k16_a_f16_18[1]), "r"(_mma_sync_m16n8k16_a_f16_18[2]), "r"(_mma_sync_m16n8k16_a_f16_18[3]), "r"(_mma_sync_m16n8k16_b_36[0]), "r"(_mma_sync_m16n8k16_b_36[1]));
                uint8_t scale_byte1_29 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_24 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_74;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_27 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_29)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_74) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_75;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_27 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_29)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_75) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_37[2];
                _mma_sync_m16n8k16_b_37[0] = _fp4_dequant_x2_74;
                _mma_sync_m16n8k16_b_37[1] = _fp4_dequant_x2_75;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_18[0]), "r"(_mma_sync_m16n8k16_a_f16_18[1]), "r"(_mma_sync_m16n8k16_a_f16_18[2]), "r"(_mma_sync_m16n8k16_a_f16_18[3]), "r"(_mma_sync_m16n8k16_b_37[0]), "r"(_mma_sync_m16n8k16_b_37[1]));
                int owner_half_30 = warp / 2;
                int n_region_31 = warp & 1;
                int u32_pos0_32 = lane * 2 + (unsigned int)owner_half_30;
                int packed_panel0_33 = u32_pos0_32 / 32;
                int packed_panel_word0_34 = u32_pos0_32 - packed_panel0_33 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((12 + packed_panel0_33) * 128 + packed_panel_word0_34 * 4 ^ ((12 + packed_panel0_33) * 128 + packed_panel_word0_34 * 4 >> 7 & 7) << 4))));
                int sf_linear0_35 = 192 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_35 / 4 * 4));
                int u32_pos1_36 = 64 + lane * 2 + (unsigned int)owner_half_30;
                int packed_panel1_37 = u32_pos1_36 / 32;
                int packed_panel_word1_38 = u32_pos1_36 - packed_panel1_37 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((12 + packed_panel1_37) * 128 + packed_panel_word1_38 * 4 ^ ((12 + packed_panel1_37) * 128 + packed_panel_word1_38 * 4 >> 7 & 7) << 4))));
                int sf_linear1_39 = sf_linear0_35 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_39 / 4 * 4));
                int a_group_base_40 = a_base;
                int a_k_byte_41 = (48 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_40 + (lane % 16 * 128 + (unsigned int)a_k_byte_41 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_41 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_19[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_19[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_19[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_19[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_19[3]) : "r"(a_frag[3]));
                int byte_shift_42 = n_region_31 * 16;
                uint8_t scale_byte0_43 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_35 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_76;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_42 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_43)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_76) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_77;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_42 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_43)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_77) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_38[2];
                _mma_sync_m16n8k16_b_38[0] = _fp4_dequant_x2_76;
                _mma_sync_m16n8k16_b_38[1] = _fp4_dequant_x2_77;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_19[0]), "r"(_mma_sync_m16n8k16_a_f16_19[1]), "r"(_mma_sync_m16n8k16_a_f16_19[2]), "r"(_mma_sync_m16n8k16_a_f16_19[3]), "r"(_mma_sync_m16n8k16_b_38[0]), "r"(_mma_sync_m16n8k16_b_38[1]));
                uint8_t scale_byte1_44 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_39 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_78;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_42 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_44)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_78) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_79;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_42 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_44)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_79) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_39[2];
                _mma_sync_m16n8k16_b_39[0] = _fp4_dequant_x2_78;
                _mma_sync_m16n8k16_b_39[1] = _fp4_dequant_x2_79;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_19[0]), "r"(_mma_sync_m16n8k16_a_f16_19[1]), "r"(_mma_sync_m16n8k16_a_f16_19[2]), "r"(_mma_sync_m16n8k16_a_f16_19[3]), "r"(_mma_sync_m16n8k16_b_39[0]), "r"(_mma_sync_m16n8k16_b_39[1]));
                int owner_half_45 = warp / 2;
                int n_region_46 = warp & 1;
                int u32_pos0_47 = lane * 2 + (unsigned int)owner_half_45;
                int packed_panel0_48 = u32_pos0_47 / 32;
                int packed_panel_word0_49 = u32_pos0_47 - packed_panel0_48 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((16 + packed_panel0_48) * 128 + packed_panel_word0_49 * 4 ^ ((16 + packed_panel0_48) * 128 + packed_panel_word0_49 * 4 >> 7 & 7) << 4))));
                int sf_linear0_50 = 256 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_50 / 4 * 4));
                int u32_pos1_51 = 64 + lane * 2 + (unsigned int)owner_half_45;
                int packed_panel1_52 = u32_pos1_51 / 32;
                int packed_panel_word1_53 = u32_pos1_51 - packed_panel1_52 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((16 + packed_panel1_52) * 128 + packed_panel_word1_53 * 4 ^ ((16 + packed_panel1_52) * 128 + packed_panel_word1_53 * 4 >> 7 & 7) << 4))));
                int sf_linear1_54 = sf_linear0_50 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_54 / 4 * 4));
                int a_group_base_55 = a_base + 2048;
                int a_k_byte_56 = lane / 16 * 8 * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_55 + (lane % 16 * 128 + (unsigned int)a_k_byte_56 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_56 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_20[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_20[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_20[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_20[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_20[3]) : "r"(a_frag[3]));
                int byte_shift_57 = n_region_46 * 16;
                uint8_t scale_byte0_58 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_50 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_80;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_57 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_58)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_80) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_81;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_57 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_58)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_81) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_40[2];
                _mma_sync_m16n8k16_b_40[0] = _fp4_dequant_x2_80;
                _mma_sync_m16n8k16_b_40[1] = _fp4_dequant_x2_81;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_20[0]), "r"(_mma_sync_m16n8k16_a_f16_20[1]), "r"(_mma_sync_m16n8k16_a_f16_20[2]), "r"(_mma_sync_m16n8k16_a_f16_20[3]), "r"(_mma_sync_m16n8k16_b_40[0]), "r"(_mma_sync_m16n8k16_b_40[1]));
                uint8_t scale_byte1_59 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_54 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_82;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_57 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_59)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_82) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_83;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_57 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_59)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_83) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_41[2];
                _mma_sync_m16n8k16_b_41[0] = _fp4_dequant_x2_82;
                _mma_sync_m16n8k16_b_41[1] = _fp4_dequant_x2_83;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_20[0]), "r"(_mma_sync_m16n8k16_a_f16_20[1]), "r"(_mma_sync_m16n8k16_a_f16_20[2]), "r"(_mma_sync_m16n8k16_a_f16_20[3]), "r"(_mma_sync_m16n8k16_b_41[0]), "r"(_mma_sync_m16n8k16_b_41[1]));
                int owner_half_60 = warp / 2;
                int n_region_61 = warp & 1;
                int u32_pos0_62 = lane * 2 + (unsigned int)owner_half_60;
                int packed_panel0_63 = u32_pos0_62 / 32;
                int packed_panel_word0_64 = u32_pos0_62 - packed_panel0_63 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((20 + packed_panel0_63) * 128 + packed_panel_word0_64 * 4 ^ ((20 + packed_panel0_63) * 128 + packed_panel_word0_64 * 4 >> 7 & 7) << 4))));
                int sf_linear0_65 = 320 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_65 / 4 * 4));
                int u32_pos1_66 = 64 + lane * 2 + (unsigned int)owner_half_60;
                int packed_panel1_67 = u32_pos1_66 / 32;
                int packed_panel_word1_68 = u32_pos1_66 - packed_panel1_67 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((20 + packed_panel1_67) * 128 + packed_panel_word1_68 * 4 ^ ((20 + packed_panel1_67) * 128 + packed_panel_word1_68 * 4 >> 7 & 7) << 4))));
                int sf_linear1_69 = sf_linear0_65 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_69 / 4 * 4));
                int a_group_base_70 = a_base + 2048;
                int a_k_byte_71 = (16 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_70 + (lane % 16 * 128 + (unsigned int)a_k_byte_71 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_71 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_21[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_21[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_21[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_21[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_21[3]) : "r"(a_frag[3]));
                int byte_shift_72 = n_region_61 * 16;
                uint8_t scale_byte0_73 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_65 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_84;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_72 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_73)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_84) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_85;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_72 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_73)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_85) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_42[2];
                _mma_sync_m16n8k16_b_42[0] = _fp4_dequant_x2_84;
                _mma_sync_m16n8k16_b_42[1] = _fp4_dequant_x2_85;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_21[0]), "r"(_mma_sync_m16n8k16_a_f16_21[1]), "r"(_mma_sync_m16n8k16_a_f16_21[2]), "r"(_mma_sync_m16n8k16_a_f16_21[3]), "r"(_mma_sync_m16n8k16_b_42[0]), "r"(_mma_sync_m16n8k16_b_42[1]));
                uint8_t scale_byte1_74 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_69 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_86;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_72 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_74)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_86) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_87;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_72 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_74)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_87) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_43[2];
                _mma_sync_m16n8k16_b_43[0] = _fp4_dequant_x2_86;
                _mma_sync_m16n8k16_b_43[1] = _fp4_dequant_x2_87;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_21[0]), "r"(_mma_sync_m16n8k16_a_f16_21[1]), "r"(_mma_sync_m16n8k16_a_f16_21[2]), "r"(_mma_sync_m16n8k16_a_f16_21[3]), "r"(_mma_sync_m16n8k16_b_43[0]), "r"(_mma_sync_m16n8k16_b_43[1]));
                int owner_half_75 = warp / 2;
                int n_region_76 = warp & 1;
                int u32_pos0_77 = lane * 2 + (unsigned int)owner_half_75;
                int packed_panel0_78 = u32_pos0_77 / 32;
                int packed_panel_word0_79 = u32_pos0_77 - packed_panel0_78 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((24 + packed_panel0_78) * 128 + packed_panel_word0_79 * 4 ^ ((24 + packed_panel0_78) * 128 + packed_panel_word0_79 * 4 >> 7 & 7) << 4))));
                int sf_linear0_80 = 384 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_80 / 4 * 4));
                int u32_pos1_81 = 64 + lane * 2 + (unsigned int)owner_half_75;
                int packed_panel1_82 = u32_pos1_81 / 32;
                int packed_panel_word1_83 = u32_pos1_81 - packed_panel1_82 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((24 + packed_panel1_82) * 128 + packed_panel_word1_83 * 4 ^ ((24 + packed_panel1_82) * 128 + packed_panel_word1_83 * 4 >> 7 & 7) << 4))));
                int sf_linear1_84 = sf_linear0_80 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_84 / 4 * 4));
                int a_group_base_85 = a_base + 2048;
                int a_k_byte_86 = (32 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_85 + (lane % 16 * 128 + (unsigned int)a_k_byte_86 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_86 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_22[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_22[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_22[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_22[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_22[3]) : "r"(a_frag[3]));
                int byte_shift_87 = n_region_76 * 16;
                uint8_t scale_byte0_88 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_80 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_88;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_87 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_88)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_88) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_89;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_87 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_88)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_89) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_44[2];
                _mma_sync_m16n8k16_b_44[0] = _fp4_dequant_x2_88;
                _mma_sync_m16n8k16_b_44[1] = _fp4_dequant_x2_89;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_22[0]), "r"(_mma_sync_m16n8k16_a_f16_22[1]), "r"(_mma_sync_m16n8k16_a_f16_22[2]), "r"(_mma_sync_m16n8k16_a_f16_22[3]), "r"(_mma_sync_m16n8k16_b_44[0]), "r"(_mma_sync_m16n8k16_b_44[1]));
                uint8_t scale_byte1_89 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_84 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_90;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_87 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_89)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_90) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_91;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_87 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_89)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_91) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_45[2];
                _mma_sync_m16n8k16_b_45[0] = _fp4_dequant_x2_90;
                _mma_sync_m16n8k16_b_45[1] = _fp4_dequant_x2_91;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_22[0]), "r"(_mma_sync_m16n8k16_a_f16_22[1]), "r"(_mma_sync_m16n8k16_a_f16_22[2]), "r"(_mma_sync_m16n8k16_a_f16_22[3]), "r"(_mma_sync_m16n8k16_b_45[0]), "r"(_mma_sync_m16n8k16_b_45[1]));
                int owner_half_90 = warp / 2;
                int n_region_91 = warp & 1;
                int u32_pos0_92 = lane * 2 + (unsigned int)owner_half_90;
                int packed_panel0_93 = u32_pos0_92 / 32;
                int packed_panel_word0_94 = u32_pos0_92 - packed_panel0_93 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((28 + packed_panel0_93) * 128 + packed_panel_word0_94 * 4 ^ ((28 + packed_panel0_93) * 128 + packed_panel_word0_94 * 4 >> 7 & 7) << 4))));
                int sf_linear0_95 = 448 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_95 / 4 * 4));
                int u32_pos1_96 = 64 + lane * 2 + (unsigned int)owner_half_90;
                int packed_panel1_97 = u32_pos1_96 / 32;
                int packed_panel_word1_98 = u32_pos1_96 - packed_panel1_97 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((28 + packed_panel1_97) * 128 + packed_panel_word1_98 * 4 ^ ((28 + packed_panel1_97) * 128 + packed_panel_word1_98 * 4 >> 7 & 7) << 4))));
                int sf_linear1_99 = sf_linear0_95 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_99 / 4 * 4));
                int a_group_base_100 = a_base + 2048;
                int a_k_byte_101 = (48 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_100 + (lane % 16 * 128 + (unsigned int)a_k_byte_101 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_101 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_23[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_23[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_23[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_23[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_23[3]) : "r"(a_frag[3]));
                int byte_shift_102 = n_region_91 * 16;
                uint8_t scale_byte0_103 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_95 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_92;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_102 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_103)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_92) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_93;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_102 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_103)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_93) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_46[2];
                _mma_sync_m16n8k16_b_46[0] = _fp4_dequant_x2_92;
                _mma_sync_m16n8k16_b_46[1] = _fp4_dequant_x2_93;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_23[0]), "r"(_mma_sync_m16n8k16_a_f16_23[1]), "r"(_mma_sync_m16n8k16_a_f16_23[2]), "r"(_mma_sync_m16n8k16_a_f16_23[3]), "r"(_mma_sync_m16n8k16_b_46[0]), "r"(_mma_sync_m16n8k16_b_46[1]));
                uint8_t scale_byte1_104 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_99 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_94;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_102 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_104)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_94) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_95;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_102 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_104)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_95) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_47[2];
                _mma_sync_m16n8k16_b_47[0] = _fp4_dequant_x2_94;
                _mma_sync_m16n8k16_b_47[1] = _fp4_dequant_x2_95;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_23[0]), "r"(_mma_sync_m16n8k16_a_f16_23[1]), "r"(_mma_sync_m16n8k16_a_f16_23[2]), "r"(_mma_sync_m16n8k16_a_f16_23[3]), "r"(_mma_sync_m16n8k16_b_47[0]), "r"(_mma_sync_m16n8k16_b_47[1]));
                int owner_half_105 = warp / 2;
                int n_region_106 = warp & 1;
                int u32_pos0_107 = lane * 2 + (unsigned int)owner_half_105;
                int packed_panel0_108 = u32_pos0_107 / 32;
                int packed_panel_word0_109 = u32_pos0_107 - packed_panel0_108 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((32 + packed_panel0_108) * 128 + packed_panel_word0_109 * 4 ^ ((32 + packed_panel0_108) * 128 + packed_panel_word0_109 * 4 >> 7 & 7) << 4))));
                int sf_linear0_110 = 512 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_110 / 4 * 4));
                int u32_pos1_111 = 64 + lane * 2 + (unsigned int)owner_half_105;
                int packed_panel1_112 = u32_pos1_111 / 32;
                int packed_panel_word1_113 = u32_pos1_111 - packed_panel1_112 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((32 + packed_panel1_112) * 128 + packed_panel_word1_113 * 4 ^ ((32 + packed_panel1_112) * 128 + packed_panel_word1_113 * 4 >> 7 & 7) << 4))));
                int sf_linear1_114 = sf_linear0_110 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_114 / 4 * 4));
                int a_group_base_115 = a_base + 4096;
                int a_k_byte_116 = lane / 16 * 8 * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_115 + (lane % 16 * 128 + (unsigned int)a_k_byte_116 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_116 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_24[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_24[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_24[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_24[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_24[3]) : "r"(a_frag[3]));
                int byte_shift_117 = n_region_106 * 16;
                uint8_t scale_byte0_118 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_110 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_96;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_117 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_118)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_96) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_97;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_117 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_118)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_97) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_48[2];
                _mma_sync_m16n8k16_b_48[0] = _fp4_dequant_x2_96;
                _mma_sync_m16n8k16_b_48[1] = _fp4_dequant_x2_97;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_24[0]), "r"(_mma_sync_m16n8k16_a_f16_24[1]), "r"(_mma_sync_m16n8k16_a_f16_24[2]), "r"(_mma_sync_m16n8k16_a_f16_24[3]), "r"(_mma_sync_m16n8k16_b_48[0]), "r"(_mma_sync_m16n8k16_b_48[1]));
                uint8_t scale_byte1_119 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_114 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_98;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_117 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_119)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_98) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_99;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_117 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_119)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_99) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_49[2];
                _mma_sync_m16n8k16_b_49[0] = _fp4_dequant_x2_98;
                _mma_sync_m16n8k16_b_49[1] = _fp4_dequant_x2_99;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_24[0]), "r"(_mma_sync_m16n8k16_a_f16_24[1]), "r"(_mma_sync_m16n8k16_a_f16_24[2]), "r"(_mma_sync_m16n8k16_a_f16_24[3]), "r"(_mma_sync_m16n8k16_b_49[0]), "r"(_mma_sync_m16n8k16_b_49[1]));
                int owner_half_120 = warp / 2;
                int n_region_121 = warp & 1;
                int u32_pos0_122 = lane * 2 + (unsigned int)owner_half_120;
                int packed_panel0_123 = u32_pos0_122 / 32;
                int packed_panel_word0_124 = u32_pos0_122 - packed_panel0_123 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((36 + packed_panel0_123) * 128 + packed_panel_word0_124 * 4 ^ ((36 + packed_panel0_123) * 128 + packed_panel_word0_124 * 4 >> 7 & 7) << 4))));
                int sf_linear0_125 = 576 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_125 / 4 * 4));
                int u32_pos1_126 = 64 + lane * 2 + (unsigned int)owner_half_120;
                int packed_panel1_127 = u32_pos1_126 / 32;
                int packed_panel_word1_128 = u32_pos1_126 - packed_panel1_127 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((36 + packed_panel1_127) * 128 + packed_panel_word1_128 * 4 ^ ((36 + packed_panel1_127) * 128 + packed_panel_word1_128 * 4 >> 7 & 7) << 4))));
                int sf_linear1_129 = sf_linear0_125 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_129 / 4 * 4));
                int a_group_base_130 = a_base + 4096;
                int a_k_byte_131 = (16 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_130 + (lane % 16 * 128 + (unsigned int)a_k_byte_131 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_131 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_25[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_25[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_25[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_25[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_25[3]) : "r"(a_frag[3]));
                int byte_shift_132 = n_region_121 * 16;
                uint8_t scale_byte0_133 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_125 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_100;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_132 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_133)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_100) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_101;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_132 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_133)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_101) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_50[2];
                _mma_sync_m16n8k16_b_50[0] = _fp4_dequant_x2_100;
                _mma_sync_m16n8k16_b_50[1] = _fp4_dequant_x2_101;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_25[0]), "r"(_mma_sync_m16n8k16_a_f16_25[1]), "r"(_mma_sync_m16n8k16_a_f16_25[2]), "r"(_mma_sync_m16n8k16_a_f16_25[3]), "r"(_mma_sync_m16n8k16_b_50[0]), "r"(_mma_sync_m16n8k16_b_50[1]));
                uint8_t scale_byte1_134 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_129 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_102;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_132 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_134)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_102) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_103;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_132 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_134)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_103) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_51[2];
                _mma_sync_m16n8k16_b_51[0] = _fp4_dequant_x2_102;
                _mma_sync_m16n8k16_b_51[1] = _fp4_dequant_x2_103;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_25[0]), "r"(_mma_sync_m16n8k16_a_f16_25[1]), "r"(_mma_sync_m16n8k16_a_f16_25[2]), "r"(_mma_sync_m16n8k16_a_f16_25[3]), "r"(_mma_sync_m16n8k16_b_51[0]), "r"(_mma_sync_m16n8k16_b_51[1]));
                int owner_half_135 = warp / 2;
                int n_region_136 = warp & 1;
                int u32_pos0_137 = lane * 2 + (unsigned int)owner_half_135;
                int packed_panel0_138 = u32_pos0_137 / 32;
                int packed_panel_word0_139 = u32_pos0_137 - packed_panel0_138 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((40 + packed_panel0_138) * 128 + packed_panel_word0_139 * 4 ^ ((40 + packed_panel0_138) * 128 + packed_panel_word0_139 * 4 >> 7 & 7) << 4))));
                int sf_linear0_140 = 640 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_140 / 4 * 4));
                int u32_pos1_141 = 64 + lane * 2 + (unsigned int)owner_half_135;
                int packed_panel1_142 = u32_pos1_141 / 32;
                int packed_panel_word1_143 = u32_pos1_141 - packed_panel1_142 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((40 + packed_panel1_142) * 128 + packed_panel_word1_143 * 4 ^ ((40 + packed_panel1_142) * 128 + packed_panel_word1_143 * 4 >> 7 & 7) << 4))));
                int sf_linear1_144 = sf_linear0_140 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_144 / 4 * 4));
                int a_group_base_145 = a_base + 4096;
                int a_k_byte_146 = (32 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_145 + (lane % 16 * 128 + (unsigned int)a_k_byte_146 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_146 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_26[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_26[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_26[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_26[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_26[3]) : "r"(a_frag[3]));
                int byte_shift_147 = n_region_136 * 16;
                uint8_t scale_byte0_148 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_140 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_104;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_147 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_148)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_104) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_105;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_147 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_148)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_105) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_52[2];
                _mma_sync_m16n8k16_b_52[0] = _fp4_dequant_x2_104;
                _mma_sync_m16n8k16_b_52[1] = _fp4_dequant_x2_105;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_26[0]), "r"(_mma_sync_m16n8k16_a_f16_26[1]), "r"(_mma_sync_m16n8k16_a_f16_26[2]), "r"(_mma_sync_m16n8k16_a_f16_26[3]), "r"(_mma_sync_m16n8k16_b_52[0]), "r"(_mma_sync_m16n8k16_b_52[1]));
                uint8_t scale_byte1_149 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_144 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_106;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_147 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_149)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_106) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_107;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_147 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_149)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_107) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_53[2];
                _mma_sync_m16n8k16_b_53[0] = _fp4_dequant_x2_106;
                _mma_sync_m16n8k16_b_53[1] = _fp4_dequant_x2_107;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_26[0]), "r"(_mma_sync_m16n8k16_a_f16_26[1]), "r"(_mma_sync_m16n8k16_a_f16_26[2]), "r"(_mma_sync_m16n8k16_a_f16_26[3]), "r"(_mma_sync_m16n8k16_b_53[0]), "r"(_mma_sync_m16n8k16_b_53[1]));
                int owner_half_150 = warp / 2;
                int n_region_151 = warp & 1;
                int u32_pos0_152 = lane * 2 + (unsigned int)owner_half_150;
                int packed_panel0_153 = u32_pos0_152 / 32;
                int packed_panel_word0_154 = u32_pos0_152 - packed_panel0_153 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((44 + packed_panel0_153) * 128 + packed_panel_word0_154 * 4 ^ ((44 + packed_panel0_153) * 128 + packed_panel_word0_154 * 4 >> 7 & 7) << 4))));
                int sf_linear0_155 = 704 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_155 / 4 * 4));
                int u32_pos1_156 = 64 + lane * 2 + (unsigned int)owner_half_150;
                int packed_panel1_157 = u32_pos1_156 / 32;
                int packed_panel_word1_158 = u32_pos1_156 - packed_panel1_157 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((44 + packed_panel1_157) * 128 + packed_panel_word1_158 * 4 ^ ((44 + packed_panel1_157) * 128 + packed_panel_word1_158 * 4 >> 7 & 7) << 4))));
                int sf_linear1_159 = sf_linear0_155 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_159 / 4 * 4));
                int a_group_base_160 = a_base + 4096;
                int a_k_byte_161 = (48 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_160 + (lane % 16 * 128 + (unsigned int)a_k_byte_161 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_161 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_27[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_27[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_27[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_27[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_27[3]) : "r"(a_frag[3]));
                int byte_shift_162 = n_region_151 * 16;
                uint8_t scale_byte0_163 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_155 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_108;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_162 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_163)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_108) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_109;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_162 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_163)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_109) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_54[2];
                _mma_sync_m16n8k16_b_54[0] = _fp4_dequant_x2_108;
                _mma_sync_m16n8k16_b_54[1] = _fp4_dequant_x2_109;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_27[0]), "r"(_mma_sync_m16n8k16_a_f16_27[1]), "r"(_mma_sync_m16n8k16_a_f16_27[2]), "r"(_mma_sync_m16n8k16_a_f16_27[3]), "r"(_mma_sync_m16n8k16_b_54[0]), "r"(_mma_sync_m16n8k16_b_54[1]));
                uint8_t scale_byte1_164 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_159 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_110;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_162 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_164)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_110) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_111;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_162 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_164)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_111) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_55[2];
                _mma_sync_m16n8k16_b_55[0] = _fp4_dequant_x2_110;
                _mma_sync_m16n8k16_b_55[1] = _fp4_dequant_x2_111;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_27[0]), "r"(_mma_sync_m16n8k16_a_f16_27[1]), "r"(_mma_sync_m16n8k16_a_f16_27[2]), "r"(_mma_sync_m16n8k16_a_f16_27[3]), "r"(_mma_sync_m16n8k16_b_55[0]), "r"(_mma_sync_m16n8k16_b_55[1]));
                int owner_half_165 = warp / 2;
                int n_region_166 = warp & 1;
                int u32_pos0_167 = lane * 2 + (unsigned int)owner_half_165;
                int packed_panel0_168 = u32_pos0_167 / 32;
                int packed_panel_word0_169 = u32_pos0_167 - packed_panel0_168 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((48 + packed_panel0_168) * 128 + packed_panel_word0_169 * 4 ^ ((48 + packed_panel0_168) * 128 + packed_panel_word0_169 * 4 >> 7 & 7) << 4))));
                int sf_linear0_170 = 768 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_170 / 4 * 4));
                int u32_pos1_171 = 64 + lane * 2 + (unsigned int)owner_half_165;
                int packed_panel1_172 = u32_pos1_171 / 32;
                int packed_panel_word1_173 = u32_pos1_171 - packed_panel1_172 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((48 + packed_panel1_172) * 128 + packed_panel_word1_173 * 4 ^ ((48 + packed_panel1_172) * 128 + packed_panel_word1_173 * 4 >> 7 & 7) << 4))));
                int sf_linear1_174 = sf_linear0_170 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_174 / 4 * 4));
                int a_group_base_175 = a_base + 6144;
                int a_k_byte_176 = lane / 16 * 8 * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_175 + (lane % 16 * 128 + (unsigned int)a_k_byte_176 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_176 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_28[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_28[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_28[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_28[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_28[3]) : "r"(a_frag[3]));
                int byte_shift_177 = n_region_166 * 16;
                uint8_t scale_byte0_178 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_170 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_112;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_177 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_178)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_112) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_113;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_177 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_178)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_113) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_56[2];
                _mma_sync_m16n8k16_b_56[0] = _fp4_dequant_x2_112;
                _mma_sync_m16n8k16_b_56[1] = _fp4_dequant_x2_113;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_28[0]), "r"(_mma_sync_m16n8k16_a_f16_28[1]), "r"(_mma_sync_m16n8k16_a_f16_28[2]), "r"(_mma_sync_m16n8k16_a_f16_28[3]), "r"(_mma_sync_m16n8k16_b_56[0]), "r"(_mma_sync_m16n8k16_b_56[1]));
                uint8_t scale_byte1_179 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_174 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_114;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_177 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_179)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_114) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_115;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_177 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_179)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_115) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_57[2];
                _mma_sync_m16n8k16_b_57[0] = _fp4_dequant_x2_114;
                _mma_sync_m16n8k16_b_57[1] = _fp4_dequant_x2_115;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_28[0]), "r"(_mma_sync_m16n8k16_a_f16_28[1]), "r"(_mma_sync_m16n8k16_a_f16_28[2]), "r"(_mma_sync_m16n8k16_a_f16_28[3]), "r"(_mma_sync_m16n8k16_b_57[0]), "r"(_mma_sync_m16n8k16_b_57[1]));
                int owner_half_180 = warp / 2;
                int n_region_181 = warp & 1;
                int u32_pos0_182 = lane * 2 + (unsigned int)owner_half_180;
                int packed_panel0_183 = u32_pos0_182 / 32;
                int packed_panel_word0_184 = u32_pos0_182 - packed_panel0_183 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((52 + packed_panel0_183) * 128 + packed_panel_word0_184 * 4 ^ ((52 + packed_panel0_183) * 128 + packed_panel_word0_184 * 4 >> 7 & 7) << 4))));
                int sf_linear0_185 = 832 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_185 / 4 * 4));
                int u32_pos1_186 = 64 + lane * 2 + (unsigned int)owner_half_180;
                int packed_panel1_187 = u32_pos1_186 / 32;
                int packed_panel_word1_188 = u32_pos1_186 - packed_panel1_187 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((52 + packed_panel1_187) * 128 + packed_panel_word1_188 * 4 ^ ((52 + packed_panel1_187) * 128 + packed_panel_word1_188 * 4 >> 7 & 7) << 4))));
                int sf_linear1_189 = sf_linear0_185 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_189 / 4 * 4));
                int a_group_base_190 = a_base + 6144;
                int a_k_byte_191 = (16 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_190 + (lane % 16 * 128 + (unsigned int)a_k_byte_191 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_191 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_29[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_29[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_29[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_29[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_29[3]) : "r"(a_frag[3]));
                int byte_shift_192 = n_region_181 * 16;
                uint8_t scale_byte0_193 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_185 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_116;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_192 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_193)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_116) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_117;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_192 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_193)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_117) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_58[2];
                _mma_sync_m16n8k16_b_58[0] = _fp4_dequant_x2_116;
                _mma_sync_m16n8k16_b_58[1] = _fp4_dequant_x2_117;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_29[0]), "r"(_mma_sync_m16n8k16_a_f16_29[1]), "r"(_mma_sync_m16n8k16_a_f16_29[2]), "r"(_mma_sync_m16n8k16_a_f16_29[3]), "r"(_mma_sync_m16n8k16_b_58[0]), "r"(_mma_sync_m16n8k16_b_58[1]));
                uint8_t scale_byte1_194 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_189 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_118;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_192 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_194)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_118) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_119;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_192 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_194)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_119) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_59[2];
                _mma_sync_m16n8k16_b_59[0] = _fp4_dequant_x2_118;
                _mma_sync_m16n8k16_b_59[1] = _fp4_dequant_x2_119;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_29[0]), "r"(_mma_sync_m16n8k16_a_f16_29[1]), "r"(_mma_sync_m16n8k16_a_f16_29[2]), "r"(_mma_sync_m16n8k16_a_f16_29[3]), "r"(_mma_sync_m16n8k16_b_59[0]), "r"(_mma_sync_m16n8k16_b_59[1]));
                int owner_half_195 = warp / 2;
                int n_region_196 = warp & 1;
                int u32_pos0_197 = lane * 2 + (unsigned int)owner_half_195;
                int packed_panel0_198 = u32_pos0_197 / 32;
                int packed_panel_word0_199 = u32_pos0_197 - packed_panel0_198 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((56 + packed_panel0_198) * 128 + packed_panel_word0_199 * 4 ^ ((56 + packed_panel0_198) * 128 + packed_panel_word0_199 * 4 >> 7 & 7) << 4))));
                int sf_linear0_200 = 896 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_200 / 4 * 4));
                int u32_pos1_201 = 64 + lane * 2 + (unsigned int)owner_half_195;
                int packed_panel1_202 = u32_pos1_201 / 32;
                int packed_panel_word1_203 = u32_pos1_201 - packed_panel1_202 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((56 + packed_panel1_202) * 128 + packed_panel_word1_203 * 4 ^ ((56 + packed_panel1_202) * 128 + packed_panel_word1_203 * 4 >> 7 & 7) << 4))));
                int sf_linear1_204 = sf_linear0_200 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_204 / 4 * 4));
                int a_group_base_205 = a_base + 6144;
                int a_k_byte_206 = (32 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_205 + (lane % 16 * 128 + (unsigned int)a_k_byte_206 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_206 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_30[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_30[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_30[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_30[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_30[3]) : "r"(a_frag[3]));
                int byte_shift_207 = n_region_196 * 16;
                uint8_t scale_byte0_208 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_200 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_120;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_207 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_208)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_120) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_121;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_207 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_208)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_121) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_60[2];
                _mma_sync_m16n8k16_b_60[0] = _fp4_dequant_x2_120;
                _mma_sync_m16n8k16_b_60[1] = _fp4_dequant_x2_121;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_30[0]), "r"(_mma_sync_m16n8k16_a_f16_30[1]), "r"(_mma_sync_m16n8k16_a_f16_30[2]), "r"(_mma_sync_m16n8k16_a_f16_30[3]), "r"(_mma_sync_m16n8k16_b_60[0]), "r"(_mma_sync_m16n8k16_b_60[1]));
                uint8_t scale_byte1_209 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_204 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_122;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_207 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_209)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_122) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_123;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_207 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_209)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_123) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_61[2];
                _mma_sync_m16n8k16_b_61[0] = _fp4_dequant_x2_122;
                _mma_sync_m16n8k16_b_61[1] = _fp4_dequant_x2_123;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_30[0]), "r"(_mma_sync_m16n8k16_a_f16_30[1]), "r"(_mma_sync_m16n8k16_a_f16_30[2]), "r"(_mma_sync_m16n8k16_a_f16_30[3]), "r"(_mma_sync_m16n8k16_b_61[0]), "r"(_mma_sync_m16n8k16_b_61[1]));
                int owner_half_210 = warp / 2;
                int n_region_211 = warp & 1;
                int u32_pos0_212 = lane * 2 + (unsigned int)owner_half_210;
                int packed_panel0_213 = u32_pos0_212 / 32;
                int packed_panel_word0_214 = u32_pos0_212 - packed_panel0_213 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw0[0])) : "r"((b_base + ((60 + packed_panel0_213) * 128 + packed_panel_word0_214 * 4 ^ ((60 + packed_panel0_213) * 128 + packed_panel_word0_214 * 4 >> 7 & 7) << 4))));
                int sf_linear0_215 = 960 + base_n + warp * 16;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word0[0])) : "r"(sf_base + sf_linear0_215 / 4 * 4));
                int u32_pos1_216 = 64 + lane * 2 + (unsigned int)owner_half_210;
                int packed_panel1_217 = u32_pos1_216 / 32;
                int packed_panel_word1_218 = u32_pos1_216 - packed_panel1_217 * 32;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw1[0])) : "r"((b_base + ((60 + packed_panel1_217) * 128 + packed_panel_word1_218 * 4 ^ ((60 + packed_panel1_217) * 128 + packed_panel_word1_218 * 4 >> 7 & 7) << 4))));
                int sf_linear1_219 = sf_linear0_215 + 8;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word1[0])) : "r"(sf_base + sf_linear1_219 / 4 * 4));
                int a_group_base_220 = a_base + 6144;
                int a_k_byte_221 = (48 + lane / 16 * 8) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(((unsigned int)a_group_base_220 + (lane % 16 * 128 + (unsigned int)a_k_byte_221 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_221 >> 7 & 7) << 4)))
                    : "memory");
                uint32_t _mma_sync_m16n8k16_a_f16_31[4];
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_31[0]) : "r"(a_frag[0]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_31[1]) : "r"(a_frag[1]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_31[2]) : "r"(a_frag[2]));
                asm(
                    "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                    "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                    "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                    "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                    "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                    "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                    : "=r"(_mma_sync_m16n8k16_a_f16_31[3]) : "r"(a_frag[3]));
                int byte_shift_222 = n_region_211 * 16;
                uint8_t scale_byte0_223 = (uint8_t)(scale_word0[0] >> (unsigned int)((sf_linear0_215 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_124;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)byte_shift_222 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_223)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_124) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_125;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw0[0] >> (unsigned int)(byte_shift_222 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte0_223)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_125) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_62[2];
                _mma_sync_m16n8k16_b_62[0] = _fp4_dequant_x2_124;
                _mma_sync_m16n8k16_b_62[1] = _fp4_dequant_x2_125;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_31[0]), "r"(_mma_sync_m16n8k16_a_f16_31[1]), "r"(_mma_sync_m16n8k16_a_f16_31[2]), "r"(_mma_sync_m16n8k16_a_f16_31[3]), "r"(_mma_sync_m16n8k16_b_62[0]), "r"(_mma_sync_m16n8k16_b_62[1]));
                uint8_t scale_byte1_224 = (uint8_t)(scale_word1[0] >> (unsigned int)((sf_linear1_219 & 3) * 8) & 255);
                uint32_t _fp4_dequant_x2_126;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)byte_shift_222 & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_224)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_126) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _fp4_dequant_x2_127;
                {
                    uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw1[0] >> (unsigned int)(byte_shift_222 + 8) & 255)) & 0xFFu);
                    uint32_t _fp4_x16x2;
                    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                        : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
                    uint32_t _scale_byte = ((uint32_t)(scale_byte1_224)) & 0xFFu;
                    uint32_t _scale_f16x2;
                    asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
                    uint32_t _scale_x16x2 = _scale_f16x2;
                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_127) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
                }
                uint32_t _mma_sync_m16n8k16_b_63[2];
                _mma_sync_m16n8k16_b_63[0] = _fp4_dequant_x2_126;
                _mma_sync_m16n8k16_b_63[1] = _fp4_dequant_x2_127;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                    : "r"(_mma_sync_m16n8k16_a_f16_31[0]), "r"(_mma_sync_m16n8k16_a_f16_31[1]), "r"(_mma_sync_m16n8k16_a_f16_31[2]), "r"(_mma_sync_m16n8k16_a_f16_31[3]), "r"(_mma_sync_m16n8k16_b_63[0]), "r"(_mma_sync_m16n8k16_b_63[1]));
            }
            __syncthreads();
            int k_tiles_1 = (K + 256 - 1) / 256;
            bool stage_valid_2 = k_tiles_1 > local_kt + 2;
            int _min_30 = ((local_kt + 2) < (k_tiles_1 - 1) ? (local_kt + 2) : (k_tiles_1 - 1));
            int safe_local_kt_3 = _min_30;
            int total_k_groups_4 = K / 16;
            int chunk_5 = tid;
            int local_m_6 = chunk_5 / 32;
            int local_k_7 = chunk_5 % 32 * 8;
            int global_m_8 = off_m + local_m_6;
            int global_k_9 = safe_local_kt_3 * 256 + local_k_7;
            int _min_31 = ((global_m_8) < (M - 1) ? (global_m_8) : (M - 1));
            int safe_m_10 = _min_31;
            int _min_32 = ((global_k_9) < (K - 8) ? (global_k_9) : (K - 8));
            int safe_k_11 = _min_32;
            bool valid_a_12 = stage_valid_2 && global_m_8 < M && global_k_9 < K;
            int a_plane_13 = local_k_7 / 64;
            int a_plane_k_14 = local_k_7 - a_plane_13 * 64;
            asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                :: "r"((v41_cute_k256_p2_a_addr + (unsigned int)(stage * 17408) + (unsigned int)(a_plane_13 * 16 * 64 * 2) + (unsigned int)(local_m_6 * 128 + a_plane_k_14 * 2 ^ (local_m_6 * 128 + a_plane_k_14 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_10 * K + safe_k_11)), "r"((valid_a_12) ? 16 : 0));
            int chunk_15 = tid + 128;
            int local_m_16 = chunk_15 / 32;
            int local_k_17 = chunk_15 % 32 * 8;
            int global_m_18 = off_m + local_m_16;
            int global_k_19 = safe_local_kt_3 * 256 + local_k_17;
            int _min_33 = ((global_m_18) < (M - 1) ? (global_m_18) : (M - 1));
            int safe_m_20 = _min_33;
            int _min_34 = ((global_k_19) < (K - 8) ? (global_k_19) : (K - 8));
            int safe_k_21 = _min_34;
            bool valid_a_22 = stage_valid_2 && global_m_18 < M && global_k_19 < K;
            int a_plane_23 = local_k_17 / 64;
            int a_plane_k_24 = local_k_17 - a_plane_23 * 64;
            asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                :: "r"((v41_cute_k256_p2_a_addr + (unsigned int)(stage * 17408) + (unsigned int)(a_plane_23 * 16 * 64 * 2) + (unsigned int)(local_m_16 * 128 + a_plane_k_24 * 2 ^ (local_m_16 * 128 + a_plane_k_24 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_20 * K + safe_k_21)), "r"((valid_a_22) ? 16 : 0));
            int chunk_25 = tid + 256;
            int local_m_26 = chunk_25 / 32;
            int local_k_27 = chunk_25 % 32 * 8;
            int global_m_28 = off_m + local_m_26;
            int global_k_29 = safe_local_kt_3 * 256 + local_k_27;
            int _min_35 = ((global_m_28) < (M - 1) ? (global_m_28) : (M - 1));
            int safe_m_30 = _min_35;
            int _min_36 = ((global_k_29) < (K - 8) ? (global_k_29) : (K - 8));
            int safe_k_31 = _min_36;
            bool valid_a_32 = stage_valid_2 && global_m_28 < M && global_k_29 < K;
            int a_plane_33 = local_k_27 / 64;
            int a_plane_k_34 = local_k_27 - a_plane_33 * 64;
            asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                :: "r"((v41_cute_k256_p2_a_addr + (unsigned int)(stage * 17408) + (unsigned int)(a_plane_33 * 16 * 64 * 2) + (unsigned int)(local_m_26 * 128 + a_plane_k_34 * 2 ^ (local_m_26 * 128 + a_plane_k_34 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_30 * K + safe_k_31)), "r"((valid_a_32) ? 16 : 0));
            int chunk_35 = tid + 384;
            int local_m_36 = chunk_35 / 32;
            int local_k_37 = chunk_35 % 32 * 8;
            int global_m_38 = off_m + local_m_36;
            int global_k_39 = safe_local_kt_3 * 256 + local_k_37;
            int _min_37 = ((global_m_38) < (M - 1) ? (global_m_38) : (M - 1));
            int safe_m_40 = _min_37;
            int _min_38 = ((global_k_39) < (K - 8) ? (global_k_39) : (K - 8));
            int safe_k_41 = _min_38;
            bool valid_a_42 = stage_valid_2 && global_m_38 < M && global_k_39 < K;
            int a_plane_43 = local_k_37 / 64;
            int a_plane_k_44 = local_k_37 - a_plane_43 * 64;
            asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                :: "r"((v41_cute_k256_p2_a_addr + (unsigned int)(stage * 17408) + (unsigned int)(a_plane_43 * 16 * 64 * 2) + (unsigned int)(local_m_36 * 128 + a_plane_k_44 * 2 ^ (local_m_36 * 128 + a_plane_k_44 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_40 * K + safe_k_41)), "r"((valid_a_42) ? 16 : 0));
            int chunk_45 = tid;
            int local_k_group_46 = chunk_45 / 32;
            int local_word_47 = chunk_45 % 32 * 4;
            int packed_panel_48 = local_word_47 / 32;
            int packed_panel_word_49 = local_word_47 - packed_panel_48 * 32;
            int packed_row_50 = local_k_group_46 * 4 + packed_panel_48;
            int global_k_group_51 = safe_local_kt_3 * 16 + local_k_group_46;
            int _min_39 = ((global_k_group_51) < (total_k_groups_4 - 1) ? (global_k_group_51) : (total_k_groups_4 - 1));
            int safe_k_group_52 = _min_39;
            bool valid_b_53 = stage_valid_2 && global_k_group_51 < total_k_groups_4;
            asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
                :: "r"((v41_cute_k256_p2_b_addr + (unsigned int)(stage * 17408) + (unsigned int)(packed_row_50 * 128 + packed_panel_word_49 * 4 ^ (packed_row_50 * 128 + packed_panel_word_49 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_52 * (N * 2) + packed_off_n * 2 + local_word_47)), "r"((valid_b_53) ? 16 : 0));
            int chunk_54 = tid + 128;
            int local_k_group_55 = chunk_54 / 32;
            int local_word_56 = chunk_54 % 32 * 4;
            int packed_panel_57 = local_word_56 / 32;
            int packed_panel_word_58 = local_word_56 - packed_panel_57 * 32;
            int packed_row_59 = local_k_group_55 * 4 + packed_panel_57;
            int global_k_group_60 = safe_local_kt_3 * 16 + local_k_group_55;
            int _min_40 = ((global_k_group_60) < (total_k_groups_4 - 1) ? (global_k_group_60) : (total_k_groups_4 - 1));
            int safe_k_group_61 = _min_40;
            bool valid_b_62 = stage_valid_2 && global_k_group_60 < total_k_groups_4;
            asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
                :: "r"((v41_cute_k256_p2_b_addr + (unsigned int)(stage * 17408) + (unsigned int)(packed_row_59 * 128 + packed_panel_word_58 * 4 ^ (packed_row_59 * 128 + packed_panel_word_58 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_61 * (N * 2) + packed_off_n * 2 + local_word_56)), "r"((valid_b_62) ? 16 : 0));
            int chunk_64 = tid + 256;
            int local_k_group_65 = chunk_64 / 32;
            int local_word_66 = chunk_64 % 32 * 4;
            int packed_panel_67 = local_word_66 / 32;
            int packed_panel_word_68 = local_word_66 - packed_panel_67 * 32;
            int packed_row_69 = local_k_group_65 * 4 + packed_panel_67;
            int global_k_group_70 = safe_local_kt_3 * 16 + local_k_group_65;
            int _min_41 = ((global_k_group_70) < (total_k_groups_4 - 1) ? (global_k_group_70) : (total_k_groups_4 - 1));
            int safe_k_group_71 = _min_41;
            bool valid_b_72 = stage_valid_2 && global_k_group_70 < total_k_groups_4;
            asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
                :: "r"((v41_cute_k256_p2_b_addr + (unsigned int)(stage * 17408) + (unsigned int)(packed_row_69 * 128 + packed_panel_word_68 * 4 ^ (packed_row_69 * 128 + packed_panel_word_68 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_71 * (N * 2) + packed_off_n * 2 + local_word_66)), "r"((valid_b_72) ? 16 : 0));
            int chunk_74 = tid + 384;
            int local_k_group_75 = chunk_74 / 32;
            int local_word_76 = chunk_74 % 32 * 4;
            int packed_panel_77 = local_word_76 / 32;
            int packed_panel_word_78 = local_word_76 - packed_panel_77 * 32;
            int packed_row_79 = local_k_group_75 * 4 + packed_panel_77;
            int global_k_group_80 = safe_local_kt_3 * 16 + local_k_group_75;
            int _min_42 = ((global_k_group_80) < (total_k_groups_4 - 1) ? (global_k_group_80) : (total_k_groups_4 - 1));
            int safe_k_group_81 = _min_42;
            bool valid_b_82 = stage_valid_2 && global_k_group_80 < total_k_groups_4;
            asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16, %2;"
                :: "r"((v41_cute_k256_p2_b_addr + (unsigned int)(stage * 17408) + (unsigned int)(packed_row_79 * 128 + packed_panel_word_78 * 4 ^ (packed_row_79 * 128 + packed_panel_word_78 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_81 * (N * 2) + packed_off_n * 2 + local_word_76)), "r"((valid_b_82) ? 16 : 0));
            int _min_43 = (((int)lane) < (15) ? ((int)lane) : (15));
            int scale_chunk_83 = warp * 16 + _min_43;
            int local_scale_k_group_84 = scale_chunk_83 / 4;
            int local_scale_n_85 = scale_chunk_83 % 4 * 16;
            int global_scale_k_group_86 = safe_local_kt_3 * 16 + local_scale_k_group_84;
            int _min_44 = ((global_scale_k_group_86) < (total_k_groups_4 - 1) ? (global_scale_k_group_86) : (total_k_groups_4 - 1));
            int safe_scale_k_group_87 = _min_44;
            bool valid_scale_88 = stage_valid_2 && global_scale_k_group_86 < total_k_groups_4;
            if (lane < 16) {
                asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                    :: "r"(v41_cute_k256_p2_scale_addr + (unsigned int)(stage * 17408) + (unsigned int)(scale_chunk_83 * 16)), "l"(B_descale + (safe_scale_k_group_87 * N + off_n + local_scale_n_85)), "r"((valid_scale_88) ? 16 : 0));
            }
            asm volatile("cp.async.commit_group;");
        }
        asm volatile("cp.async.wait_group 0;");
        float alpha_value = 1.0f;
        {
            alpha_value = alpha[0];
        }
        int row0 = lane / 4;
        int row1 = row0 + 8;
        int output_partition = warp * 16;
        int col_pair = (unsigned int)(off_n + output_partition) + lane % 4 * 2;
        if (off_m + row0 < M && col_pair < N) {
            long long row0_base = (long long)(off_m + row0) * (long long)N + (long long)col_pair;
            {
                const float2 _prescale2_0 = {alpha_value, alpha_value};
                #if __CUDA_ARCH__ >= 1000
                #pragma unroll
                for (int _ps = 0; _ps < 1; _ps++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc0[0])[_ps], _prescale2_0);
                #else
                #pragma unroll
                for (int _ps = 0; _ps < 2; _ps++)
                    acc0[0 + _ps] *= alpha_value;
                #endif
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc0[0 + 0], acc0[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + row0_base))[0]) = _pk;
            }
            if (col_pair + 8 < N) {
                {
                    const float2 _prescale2_1 = {alpha_value, alpha_value};
                    #if __CUDA_ARCH__ >= 1000
                    #pragma unroll
                    for (int _ps = 0; _ps < 1; _ps++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc1[0])[_ps], _prescale2_1);
                    #else
                    #pragma unroll
                    for (int _ps = 0; _ps < 2; _ps++)
                        acc1[0 + _ps] *= alpha_value;
                    #endif
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(acc1[0 + 0], acc1[0 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + (row0_base + 8)))[0]) = _pk;
                }
            }
        }
        if (off_m + row1 < M && col_pair < N) {
            long long row1_base = (long long)(off_m + row1) * (long long)N + (long long)col_pair;
            {
                const float2 _prescale2_2 = {alpha_value, alpha_value};
                #if __CUDA_ARCH__ >= 1000
                #pragma unroll
                for (int _ps = 0; _ps < 1; _ps++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc0[2])[_ps], _prescale2_2);
                #else
                #pragma unroll
                for (int _ps = 0; _ps < 2; _ps++)
                    acc0[2 + _ps] *= alpha_value;
                #endif
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc0[2 + 0], acc0[2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + row1_base))[0]) = _pk;
            }
            if (col_pair + 8 < N) {
                {
                    const float2 _prescale2_3 = {alpha_value, alpha_value};
                    #if __CUDA_ARCH__ >= 1000
                    #pragma unroll
                    for (int _ps = 0; _ps < 1; _ps++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc1[2])[_ps], _prescale2_3);
                    #else
                    #pragma unroll
                    for (int _ps = 0; _ps < 2; _ps++)
                        acc1[2 + _ps] *= alpha_value;
                    #endif
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(acc1[2 + 0], acc1[2 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + (row1_base + 8)))[0]) = _pk;
                }
            }
        }
    }
    {
        __syncthreads();
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"
