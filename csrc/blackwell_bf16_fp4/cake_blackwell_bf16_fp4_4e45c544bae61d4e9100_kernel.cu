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
#define SMEM_CUTE_REMAINING_SMALL_A_OFF 0
#define SMEM_CUTE_REMAINING_SMALL_A_STAGE_BYTES 4096
#define SMEM_CUTE_REMAINING_SMALL_A_STRIDE 7168
#define SMEM_CUTE_REMAINING_SMALL_B_OFF 4096
#define SMEM_CUTE_REMAINING_SMALL_B_STAGE_BYTES 2048
#define SMEM_CUTE_REMAINING_SMALL_B_STRIDE 7168
#define SMEM_CUTE_REMAINING_SMALL_SCALE_OFF 6144
#define SMEM_CUTE_REMAINING_SMALL_SCALE_STAGE_BYTES 256
#define SMEM_CUTE_REMAINING_SMALL_SCALE_STRIDE 7168
#define SMEM_TOTAL 21504
#define THREADS 128
#define HAS_ALPHA 1
#define ENABLE_PDL 1

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
kernel_cake_blackwell_bf16_fp4_4e45c544bae61d4e9100(__nv_bfloat16* __restrict__ A, int* __restrict__ B, uint8_t* __restrict__ B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K)
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
    __nv_bfloat16* cute_remaining_small_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int cute_remaining_small_a_addr = smem + 0;
    int* cute_remaining_small_b = reinterpret_cast<int*>(smem_raw + 4096);
    const int cute_remaining_small_b_addr = smem + 4096;
    uint8_t* cute_remaining_small_scale = reinterpret_cast<uint8_t*>(smem_raw + 6144);
    const int cute_remaining_small_scale_addr = smem + 6144;

    // === Task calls (dependency order) ===
    int grid_n = 2 * ((N + 64 - 1) / 64);
    int work = blockIdx.x;
    int tile_m = work / grid_n;
    int logical_tile_n = work - tile_m * grid_n;
    int tile_n = logical_tile_n / 2;
    int n_half = logical_tile_n & 1;
    int virtual_warp = n_half * 2 + warp / 2;
    int logical_n_warp = warp & 1;
    int off_m = tile_m * 16;
    int off_n = tile_n * 64;
    int packed_off_n = tile_n * 64;
    int k_tiles = (K + 128 - 1) / 128;
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int k_tiles_0 = (K + 128 - 1) / 128;
    bool stage_valid = k_tiles_0 > 0;
    int _min_0 = ((0) < (k_tiles_0 - 1) ? (0) : (k_tiles_0 - 1));
    int safe_local_kt = _min_0;
    int total_k_groups = K / 16;
    int chunk = tid;
    int local_m = chunk / 16;
    int local_k = chunk % 16 * 8;
    int global_m = off_m + local_m;
    int global_k = safe_local_kt * 128 + local_k;
    int _min_1 = ((global_m) < (M - 1) ? (global_m) : (M - 1));
    int safe_m = _min_1;
    int _min_2 = ((global_k) < (K - 8) ? (global_k) : (K - 8));
    int safe_k = _min_2;
    bool valid_a = stage_valid && global_m < M && global_k < K;
    int a_plane = local_k / 64;
    int a_plane_k = local_k - a_plane * 64;
    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
        :: "r"((cute_remaining_small_a_addr + (unsigned int)(a_plane * 16 * 64 * 2) + (unsigned int)(local_m * 128 + a_plane_k * 2 ^ (local_m * 128 + a_plane_k * 2 >> 7 & 7) << 4))), "l"(A + (safe_m * K + safe_k)), "r"((valid_a) ? 16 : 0));
    int chunk_1 = tid + 128;
    int local_m_2 = chunk_1 / 16;
    int local_k_3 = chunk_1 % 16 * 8;
    int global_m_4 = off_m + local_m_2;
    int global_k_5 = safe_local_kt * 128 + local_k_3;
    int _min_3 = ((global_m_4) < (M - 1) ? (global_m_4) : (M - 1));
    int safe_m_6 = _min_3;
    int _min_4 = ((global_k_5) < (K - 8) ? (global_k_5) : (K - 8));
    int safe_k_7 = _min_4;
    bool valid_a_8 = stage_valid && global_m_4 < M && global_k_5 < K;
    int a_plane_9 = local_k_3 / 64;
    int a_plane_k_10 = local_k_3 - a_plane_9 * 64;
    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
        :: "r"((cute_remaining_small_a_addr + (unsigned int)(a_plane_9 * 16 * 64 * 2) + (unsigned int)(local_m_2 * 128 + a_plane_k_10 * 2 ^ (local_m_2 * 128 + a_plane_k_10 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_6 * K + safe_k_7)), "r"((valid_a_8) ? 16 : 0));
    int chunk_11 = tid;
    int local_k_group = chunk_11 / 64;
    int local_word = chunk_11 % 64;
    int packed_panel = local_word / 32;
    int packed_panel_word = local_word - packed_panel * 32;
    int packed_row = local_k_group * 2 + packed_panel;
    int global_k_group = safe_local_kt * 8 + local_k_group;
    int _min_5 = ((global_k_group) < (total_k_groups - 1) ? (global_k_group) : (total_k_groups - 1));
    int safe_k_group = _min_5;
    bool valid_b = stage_valid && global_k_group < total_k_groups;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + (unsigned int)(packed_row * 128 + packed_panel_word * 4 ^ (packed_row * 128 + packed_panel_word * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group * (N * 2) + packed_off_n * 2 + local_word * 2 + n_half)), "r"((valid_b) ? 4 : 0));
    int chunk_12 = tid + 128;
    int local_k_group_13 = chunk_12 / 64;
    int local_word_14 = chunk_12 % 64;
    int packed_panel_15 = local_word_14 / 32;
    int packed_panel_word_16 = local_word_14 - packed_panel_15 * 32;
    int packed_row_17 = local_k_group_13 * 2 + packed_panel_15;
    int global_k_group_18 = safe_local_kt * 8 + local_k_group_13;
    int _min_6 = ((global_k_group_18) < (total_k_groups - 1) ? (global_k_group_18) : (total_k_groups - 1));
    int safe_k_group_19 = _min_6;
    bool valid_b_20 = stage_valid && global_k_group_18 < total_k_groups;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + (unsigned int)(packed_row_17 * 128 + packed_panel_word_16 * 4 ^ (packed_row_17 * 128 + packed_panel_word_16 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_19 * (N * 2) + packed_off_n * 2 + local_word_14 * 2 + n_half)), "r"((valid_b_20) ? 4 : 0));
    int chunk_21 = tid + 256;
    int local_k_group_22 = chunk_21 / 64;
    int local_word_23 = chunk_21 % 64;
    int packed_panel_24 = local_word_23 / 32;
    int packed_panel_word_25 = local_word_23 - packed_panel_24 * 32;
    int packed_row_26 = local_k_group_22 * 2 + packed_panel_24;
    int global_k_group_27 = safe_local_kt * 8 + local_k_group_22;
    int _min_7 = ((global_k_group_27) < (total_k_groups - 1) ? (global_k_group_27) : (total_k_groups - 1));
    int safe_k_group_28 = _min_7;
    bool valid_b_29 = stage_valid && global_k_group_27 < total_k_groups;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + (unsigned int)(packed_row_26 * 128 + packed_panel_word_25 * 4 ^ (packed_row_26 * 128 + packed_panel_word_25 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_28 * (N * 2) + packed_off_n * 2 + local_word_23 * 2 + n_half)), "r"((valid_b_29) ? 4 : 0));
    int chunk_30 = tid + 384;
    int local_k_group_31 = chunk_30 / 64;
    int local_word_32 = chunk_30 % 64;
    int packed_panel_33 = local_word_32 / 32;
    int packed_panel_word_34 = local_word_32 - packed_panel_33 * 32;
    int packed_row_35 = local_k_group_31 * 2 + packed_panel_33;
    int global_k_group_36 = safe_local_kt * 8 + local_k_group_31;
    int _min_8 = ((global_k_group_36) < (total_k_groups - 1) ? (global_k_group_36) : (total_k_groups - 1));
    int safe_k_group_37 = _min_8;
    bool valid_b_38 = stage_valid && global_k_group_36 < total_k_groups;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + (unsigned int)(packed_row_35 * 128 + packed_panel_word_34 * 4 ^ (packed_row_35 * 128 + packed_panel_word_34 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_37 * (N * 2) + packed_off_n * 2 + local_word_32 * 2 + n_half)), "r"((valid_b_38) ? 4 : 0));
    unsigned int _min_9 = ((lane) < (3) ? (lane) : (3));
    int scale_chunk = (unsigned int)(warp * 4) + _min_9;
    int local_scale_k_group = scale_chunk / 2;
    int local_scale_n = scale_chunk % 2 * 16;
    int global_scale_k_group = safe_local_kt * 8 + local_scale_k_group;
    int _min_10 = ((global_scale_k_group) < (total_k_groups - 1) ? (global_scale_k_group) : (total_k_groups - 1));
    int safe_scale_k_group = _min_10;
    bool valid_scale = stage_valid && global_scale_k_group < total_k_groups;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %0, 0;\n\t"
        "@p cp.async.cg.shared::cta.global.L2::128B [%1], [%2], 16;\n\t"
        "}"
        :: "r"((valid_scale && lane < 4) ? 1 : 0), "r"(cute_remaining_small_scale_addr + (unsigned int)(scale_chunk * 16)), "l"(B_descale + (safe_scale_k_group * N + off_n + n_half * 32 + local_scale_n)));
    asm volatile("cp.async.commit_group;");
    int k_tiles_39 = (K + 128 - 1) / 128;
    bool stage_valid_40 = k_tiles_39 > 1;
    int _min_11 = ((1) < (k_tiles_39 - 1) ? (1) : (k_tiles_39 - 1));
    int safe_local_kt_41 = _min_11;
    int total_k_groups_42 = K / 16;
    int chunk_43 = tid;
    int local_m_44 = chunk_43 / 16;
    int local_k_45 = chunk_43 % 16 * 8;
    int global_m_46 = off_m + local_m_44;
    int global_k_47 = safe_local_kt_41 * 128 + local_k_45;
    int _min_12 = ((global_m_46) < (M - 1) ? (global_m_46) : (M - 1));
    int safe_m_48 = _min_12;
    int _min_13 = ((global_k_47) < (K - 8) ? (global_k_47) : (K - 8));
    int safe_k_49 = _min_13;
    bool valid_a_50 = stage_valid_40 && global_m_46 < M && global_k_47 < K;
    int a_plane_51 = local_k_45 / 64;
    int a_plane_k_52 = local_k_45 - a_plane_51 * 64;
    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
        :: "r"((cute_remaining_small_a_addr + 7168 + (unsigned int)(a_plane_51 * 16 * 64 * 2) + (unsigned int)(local_m_44 * 128 + a_plane_k_52 * 2 ^ (local_m_44 * 128 + a_plane_k_52 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_48 * K + safe_k_49)), "r"((valid_a_50) ? 16 : 0));
    int chunk_53 = tid + 128;
    int local_m_54 = chunk_53 / 16;
    int local_k_55 = chunk_53 % 16 * 8;
    int global_m_56 = off_m + local_m_54;
    int global_k_57 = safe_local_kt_41 * 128 + local_k_55;
    int _min_14 = ((global_m_56) < (M - 1) ? (global_m_56) : (M - 1));
    int safe_m_58 = _min_14;
    int _min_15 = ((global_k_57) < (K - 8) ? (global_k_57) : (K - 8));
    int safe_k_59 = _min_15;
    bool valid_a_60 = stage_valid_40 && global_m_56 < M && global_k_57 < K;
    int a_plane_61 = local_k_55 / 64;
    int a_plane_k_62 = local_k_55 - a_plane_61 * 64;
    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
        :: "r"((cute_remaining_small_a_addr + 7168 + (unsigned int)(a_plane_61 * 16 * 64 * 2) + (unsigned int)(local_m_54 * 128 + a_plane_k_62 * 2 ^ (local_m_54 * 128 + a_plane_k_62 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_58 * K + safe_k_59)), "r"((valid_a_60) ? 16 : 0));
    int chunk_63 = tid;
    int local_k_group_64 = chunk_63 / 64;
    int local_word_65 = chunk_63 % 64;
    int packed_panel_66 = local_word_65 / 32;
    int packed_panel_word_67 = local_word_65 - packed_panel_66 * 32;
    int packed_row_68 = local_k_group_64 * 2 + packed_panel_66;
    int global_k_group_69 = safe_local_kt_41 * 8 + local_k_group_64;
    int _min_16 = ((global_k_group_69) < (total_k_groups_42 - 1) ? (global_k_group_69) : (total_k_groups_42 - 1));
    int safe_k_group_70 = _min_16;
    bool valid_b_71 = stage_valid_40 && global_k_group_69 < total_k_groups_42;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + 7168 + (unsigned int)(packed_row_68 * 128 + packed_panel_word_67 * 4 ^ (packed_row_68 * 128 + packed_panel_word_67 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_70 * (N * 2) + packed_off_n * 2 + local_word_65 * 2 + n_half)), "r"((valid_b_71) ? 4 : 0));
    int chunk_72 = tid + 128;
    int local_k_group_73 = chunk_72 / 64;
    int local_word_74 = chunk_72 % 64;
    int packed_panel_75 = local_word_74 / 32;
    int packed_panel_word_76 = local_word_74 - packed_panel_75 * 32;
    int packed_row_77 = local_k_group_73 * 2 + packed_panel_75;
    int global_k_group_78 = safe_local_kt_41 * 8 + local_k_group_73;
    int _min_17 = ((global_k_group_78) < (total_k_groups_42 - 1) ? (global_k_group_78) : (total_k_groups_42 - 1));
    int safe_k_group_79 = _min_17;
    bool valid_b_80 = stage_valid_40 && global_k_group_78 < total_k_groups_42;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + 7168 + (unsigned int)(packed_row_77 * 128 + packed_panel_word_76 * 4 ^ (packed_row_77 * 128 + packed_panel_word_76 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_79 * (N * 2) + packed_off_n * 2 + local_word_74 * 2 + n_half)), "r"((valid_b_80) ? 4 : 0));
    int chunk_81 = tid + 256;
    int local_k_group_82 = chunk_81 / 64;
    int local_word_83 = chunk_81 % 64;
    int packed_panel_84 = local_word_83 / 32;
    int packed_panel_word_85 = local_word_83 - packed_panel_84 * 32;
    int packed_row_86 = local_k_group_82 * 2 + packed_panel_84;
    int global_k_group_87 = safe_local_kt_41 * 8 + local_k_group_82;
    int _min_18 = ((global_k_group_87) < (total_k_groups_42 - 1) ? (global_k_group_87) : (total_k_groups_42 - 1));
    int safe_k_group_88 = _min_18;
    bool valid_b_89 = stage_valid_40 && global_k_group_87 < total_k_groups_42;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + 7168 + (unsigned int)(packed_row_86 * 128 + packed_panel_word_85 * 4 ^ (packed_row_86 * 128 + packed_panel_word_85 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_88 * (N * 2) + packed_off_n * 2 + local_word_83 * 2 + n_half)), "r"((valid_b_89) ? 4 : 0));
    int chunk_90 = tid + 384;
    int local_k_group_91 = chunk_90 / 64;
    int local_word_92 = chunk_90 % 64;
    int packed_panel_93 = local_word_92 / 32;
    int packed_panel_word_94 = local_word_92 - packed_panel_93 * 32;
    int packed_row_95 = local_k_group_91 * 2 + packed_panel_93;
    int global_k_group_96 = safe_local_kt_41 * 8 + local_k_group_91;
    int _min_19 = ((global_k_group_96) < (total_k_groups_42 - 1) ? (global_k_group_96) : (total_k_groups_42 - 1));
    int safe_k_group_97 = _min_19;
    bool valid_b_98 = stage_valid_40 && global_k_group_96 < total_k_groups_42;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + 7168 + (unsigned int)(packed_row_95 * 128 + packed_panel_word_94 * 4 ^ (packed_row_95 * 128 + packed_panel_word_94 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_97 * (N * 2) + packed_off_n * 2 + local_word_92 * 2 + n_half)), "r"((valid_b_98) ? 4 : 0));
    unsigned int _min_20 = ((lane) < (3) ? (lane) : (3));
    int scale_chunk_99 = (unsigned int)(warp * 4) + _min_20;
    int local_scale_k_group_100 = scale_chunk_99 / 2;
    int local_scale_n_101 = scale_chunk_99 % 2 * 16;
    int global_scale_k_group_102 = safe_local_kt_41 * 8 + local_scale_k_group_100;
    int _min_21 = ((global_scale_k_group_102) < (total_k_groups_42 - 1) ? (global_scale_k_group_102) : (total_k_groups_42 - 1));
    int safe_scale_k_group_103 = _min_21;
    bool valid_scale_104 = stage_valid_40 && global_scale_k_group_102 < total_k_groups_42;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %0, 0;\n\t"
        "@p cp.async.cg.shared::cta.global.L2::128B [%1], [%2], 16;\n\t"
        "}"
        :: "r"((valid_scale_104 && lane < 4) ? 1 : 0), "r"(cute_remaining_small_scale_addr + 7168 + (unsigned int)(scale_chunk_99 * 16)), "l"(B_descale + (safe_scale_k_group_103 * N + off_n + n_half * 32 + local_scale_n_101)));
    asm volatile("cp.async.commit_group;");
    int k_tiles_105 = (K + 128 - 1) / 128;
    bool stage_valid_106 = k_tiles_105 > 2;
    int _min_22 = ((2) < (k_tiles_105 - 1) ? (2) : (k_tiles_105 - 1));
    int safe_local_kt_107 = _min_22;
    int total_k_groups_108 = K / 16;
    int chunk_109 = tid;
    int local_m_110 = chunk_109 / 16;
    int local_k_111 = chunk_109 % 16 * 8;
    int global_m_112 = off_m + local_m_110;
    int global_k_113 = safe_local_kt_107 * 128 + local_k_111;
    int _min_23 = ((global_m_112) < (M - 1) ? (global_m_112) : (M - 1));
    int safe_m_114 = _min_23;
    int _min_24 = ((global_k_113) < (K - 8) ? (global_k_113) : (K - 8));
    int safe_k_115 = _min_24;
    bool valid_a_116 = stage_valid_106 && global_m_112 < M && global_k_113 < K;
    int a_plane_117 = local_k_111 / 64;
    int a_plane_k_118 = local_k_111 - a_plane_117 * 64;
    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
        :: "r"((cute_remaining_small_a_addr + 14336 + (unsigned int)(a_plane_117 * 16 * 64 * 2) + (unsigned int)(local_m_110 * 128 + a_plane_k_118 * 2 ^ (local_m_110 * 128 + a_plane_k_118 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_114 * K + safe_k_115)), "r"((valid_a_116) ? 16 : 0));
    int chunk_119 = tid + 128;
    int local_m_120 = chunk_119 / 16;
    int local_k_121 = chunk_119 % 16 * 8;
    int global_m_122 = off_m + local_m_120;
    int global_k_123 = safe_local_kt_107 * 128 + local_k_121;
    int _min_25 = ((global_m_122) < (M - 1) ? (global_m_122) : (M - 1));
    int safe_m_124 = _min_25;
    int _min_26 = ((global_k_123) < (K - 8) ? (global_k_123) : (K - 8));
    int safe_k_125 = _min_26;
    bool valid_a_126 = stage_valid_106 && global_m_122 < M && global_k_123 < K;
    int a_plane_127 = local_k_121 / 64;
    int a_plane_k_128 = local_k_121 - a_plane_127 * 64;
    asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
        :: "r"((cute_remaining_small_a_addr + 14336 + (unsigned int)(a_plane_127 * 16 * 64 * 2) + (unsigned int)(local_m_120 * 128 + a_plane_k_128 * 2 ^ (local_m_120 * 128 + a_plane_k_128 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_124 * K + safe_k_125)), "r"((valid_a_126) ? 16 : 0));
    int chunk_129 = tid;
    int local_k_group_130 = chunk_129 / 64;
    int local_word_131 = chunk_129 % 64;
    int packed_panel_132 = local_word_131 / 32;
    int packed_panel_word_133 = local_word_131 - packed_panel_132 * 32;
    int packed_row_134 = local_k_group_130 * 2 + packed_panel_132;
    int global_k_group_135 = safe_local_kt_107 * 8 + local_k_group_130;
    int _min_27 = ((global_k_group_135) < (total_k_groups_108 - 1) ? (global_k_group_135) : (total_k_groups_108 - 1));
    int safe_k_group_136 = _min_27;
    bool valid_b_137 = stage_valid_106 && global_k_group_135 < total_k_groups_108;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + 14336 + (unsigned int)(packed_row_134 * 128 + packed_panel_word_133 * 4 ^ (packed_row_134 * 128 + packed_panel_word_133 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_136 * (N * 2) + packed_off_n * 2 + local_word_131 * 2 + n_half)), "r"((valid_b_137) ? 4 : 0));
    int chunk_138 = tid + 128;
    int local_k_group_139 = chunk_138 / 64;
    int local_word_140 = chunk_138 % 64;
    int packed_panel_141 = local_word_140 / 32;
    int packed_panel_word_142 = local_word_140 - packed_panel_141 * 32;
    int packed_row_143 = local_k_group_139 * 2 + packed_panel_141;
    int global_k_group_144 = safe_local_kt_107 * 8 + local_k_group_139;
    int _min_28 = ((global_k_group_144) < (total_k_groups_108 - 1) ? (global_k_group_144) : (total_k_groups_108 - 1));
    int safe_k_group_145 = _min_28;
    bool valid_b_146 = stage_valid_106 && global_k_group_144 < total_k_groups_108;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + 14336 + (unsigned int)(packed_row_143 * 128 + packed_panel_word_142 * 4 ^ (packed_row_143 * 128 + packed_panel_word_142 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_145 * (N * 2) + packed_off_n * 2 + local_word_140 * 2 + n_half)), "r"((valid_b_146) ? 4 : 0));
    int chunk_147 = tid + 256;
    int local_k_group_148 = chunk_147 / 64;
    int local_word_149 = chunk_147 % 64;
    int packed_panel_150 = local_word_149 / 32;
    int packed_panel_word_151 = local_word_149 - packed_panel_150 * 32;
    int packed_row_152 = local_k_group_148 * 2 + packed_panel_150;
    int global_k_group_153 = safe_local_kt_107 * 8 + local_k_group_148;
    int _min_29 = ((global_k_group_153) < (total_k_groups_108 - 1) ? (global_k_group_153) : (total_k_groups_108 - 1));
    int safe_k_group_154 = _min_29;
    bool valid_b_155 = stage_valid_106 && global_k_group_153 < total_k_groups_108;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + 14336 + (unsigned int)(packed_row_152 * 128 + packed_panel_word_151 * 4 ^ (packed_row_152 * 128 + packed_panel_word_151 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_154 * (N * 2) + packed_off_n * 2 + local_word_149 * 2 + n_half)), "r"((valid_b_155) ? 4 : 0));
    int chunk_156 = tid + 384;
    int local_k_group_157 = chunk_156 / 64;
    int local_word_158 = chunk_156 % 64;
    int packed_panel_159 = local_word_158 / 32;
    int packed_panel_word_160 = local_word_158 - packed_panel_159 * 32;
    int packed_row_161 = local_k_group_157 * 2 + packed_panel_159;
    int global_k_group_162 = safe_local_kt_107 * 8 + local_k_group_157;
    int _min_30 = ((global_k_group_162) < (total_k_groups_108 - 1) ? (global_k_group_162) : (total_k_groups_108 - 1));
    int safe_k_group_163 = _min_30;
    bool valid_b_164 = stage_valid_106 && global_k_group_162 < total_k_groups_108;
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
        :: "r"((cute_remaining_small_b_addr + 14336 + (unsigned int)(packed_row_161 * 128 + packed_panel_word_160 * 4 ^ (packed_row_161 * 128 + packed_panel_word_160 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_163 * (N * 2) + packed_off_n * 2 + local_word_158 * 2 + n_half)), "r"((valid_b_164) ? 4 : 0));
    unsigned int _min_31 = ((lane) < (3) ? (lane) : (3));
    int scale_chunk_165 = (unsigned int)(warp * 4) + _min_31;
    int local_scale_k_group_166 = scale_chunk_165 / 2;
    int local_scale_n_167 = scale_chunk_165 % 2 * 16;
    int global_scale_k_group_168 = safe_local_kt_107 * 8 + local_scale_k_group_166;
    int _min_32 = ((global_scale_k_group_168) < (total_k_groups_108 - 1) ? (global_scale_k_group_168) : (total_k_groups_108 - 1));
    int safe_scale_k_group_169 = _min_32;
    bool valid_scale_170 = stage_valid_106 && global_scale_k_group_168 < total_k_groups_108;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %0, 0;\n\t"
        "@p cp.async.cg.shared::cta.global.L2::128B [%1], [%2], 16;\n\t"
        "}"
        :: "r"((valid_scale_170 && lane < 4) ? 1 : 0), "r"(cute_remaining_small_scale_addr + 14336 + (unsigned int)(scale_chunk_165 * 16)), "l"(B_descale + (safe_scale_k_group_169 * N + off_n + n_half * 32 + local_scale_n_167)));
    asm volatile("cp.async.commit_group;");
    unsigned int a_frag[4];
    unsigned int raw[1];
    unsigned int scale_word[1];
    float acc0[4];
    acc0[0] = 0.0f;
    acc0[1] = 0.0f;
    acc0[2] = 0.0f;
    acc0[3] = 0.0f;
    #pragma unroll 1
    for (int local_kt = 0; local_kt < k_tiles; local_kt++) {
        int stage = local_kt % 3;
        asm volatile("cp.async.wait_group 2;");
        __syncthreads();
        int a_base = cute_remaining_small_a_addr + (unsigned int)(stage * 7168);
        int b_base = cute_remaining_small_b_addr + (unsigned int)(stage * 7168);
        int sf_base = cute_remaining_small_scale_addr + (unsigned int)(stage * 7168);
        int tc_col = lane / 4;
        int base_n = tc_col;
        int a_group_base = a_base;
        int a_k_byte = lane / 16 * 8 * 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(((unsigned int)a_group_base + (lane % 16 * 128 + (unsigned int)a_k_byte ^ (lane % 16 * 128 + (unsigned int)a_k_byte >> 7 & 7) << 4)))
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
        int n_region = virtual_warp & 1;
        int u32_pos = (unsigned int)(logical_n_warp * 32) + lane;
        int packed_panel_0 = u32_pos / 32;
        int packed_panel_word_1 = u32_pos - packed_panel_0 * 32;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])) : "r"((b_base + (packed_panel_0 * 128 + packed_panel_word_1 * 4 ^ (packed_panel_0 * 128 + packed_panel_word_1 * 4 >> 7 & 7) << 4))));
        int sf_linear = base_n + logical_n_warp * 8 + warp / 2 * 16;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear / 4 * 4));
        uint8_t scale_byte = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear & 3) * 8) & 255);
        int byte_shift = n_region * 16;
        uint32_t _fp4_dequant_x2_0;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)byte_shift & 255)) & 0xFFu);
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
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)(byte_shift + 8) & 255)) & 0xFFu);
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
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "r"(_mma_sync_m16n8k16_a_f16_0[0]), "r"(_mma_sync_m16n8k16_a_f16_0[1]), "r"(_mma_sync_m16n8k16_a_f16_0[2]), "r"(_mma_sync_m16n8k16_a_f16_0[3]), "r"(_mma_sync_m16n8k16_b_0[0]), "r"(_mma_sync_m16n8k16_b_0[1]));
        int a_group_base_2 = a_base;
        int a_k_byte_3 = (16 + lane / 16 * 8) * 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(((unsigned int)a_group_base_2 + (lane % 16 * 128 + (unsigned int)a_k_byte_3 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_3 >> 7 & 7) << 4)))
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
        int n_region_4 = virtual_warp & 1;
        int u32_pos_5 = (unsigned int)(logical_n_warp * 32) + lane;
        int packed_panel_6 = u32_pos_5 / 32;
        int packed_panel_word_7 = u32_pos_5 - packed_panel_6 * 32;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])) : "r"((b_base + ((2 + packed_panel_6) * 128 + packed_panel_word_7 * 4 ^ ((2 + packed_panel_6) * 128 + packed_panel_word_7 * 4 >> 7 & 7) << 4))));
        int sf_linear_8 = 32 + base_n + logical_n_warp * 8 + warp / 2 * 16;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_8 / 4 * 4));
        uint8_t scale_byte_9 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_8 & 3) * 8) & 255);
        int byte_shift_10 = n_region_4 * 16;
        uint32_t _fp4_dequant_x2_2;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)byte_shift_10 & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_9)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_2) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _fp4_dequant_x2_3;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)(byte_shift_10 + 8) & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_9)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_3) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _mma_sync_m16n8k16_b_1[2];
        _mma_sync_m16n8k16_b_1[0] = _fp4_dequant_x2_2;
        _mma_sync_m16n8k16_b_1[1] = _fp4_dequant_x2_3;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "r"(_mma_sync_m16n8k16_a_f16_1[0]), "r"(_mma_sync_m16n8k16_a_f16_1[1]), "r"(_mma_sync_m16n8k16_a_f16_1[2]), "r"(_mma_sync_m16n8k16_a_f16_1[3]), "r"(_mma_sync_m16n8k16_b_1[0]), "r"(_mma_sync_m16n8k16_b_1[1]));
        int a_group_base_11 = a_base;
        int a_k_byte_12 = (32 + lane / 16 * 8) * 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(((unsigned int)a_group_base_11 + (lane % 16 * 128 + (unsigned int)a_k_byte_12 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_12 >> 7 & 7) << 4)))
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
        int n_region_13 = virtual_warp & 1;
        int u32_pos_14 = (unsigned int)(logical_n_warp * 32) + lane;
        int packed_panel_16 = u32_pos_14 / 32;
        int packed_panel_word_17 = u32_pos_14 - packed_panel_16 * 32;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])) : "r"((b_base + ((4 + packed_panel_16) * 128 + packed_panel_word_17 * 4 ^ ((4 + packed_panel_16) * 128 + packed_panel_word_17 * 4 >> 7 & 7) << 4))));
        int sf_linear_18 = 64 + base_n + logical_n_warp * 8 + warp / 2 * 16;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_18 / 4 * 4));
        uint8_t scale_byte_19 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_18 & 3) * 8) & 255);
        int byte_shift_20 = n_region_13 * 16;
        uint32_t _fp4_dequant_x2_4;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)byte_shift_20 & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_19)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_4) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _fp4_dequant_x2_5;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)(byte_shift_20 + 8) & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_19)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_5) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _mma_sync_m16n8k16_b_2[2];
        _mma_sync_m16n8k16_b_2[0] = _fp4_dequant_x2_4;
        _mma_sync_m16n8k16_b_2[1] = _fp4_dequant_x2_5;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "r"(_mma_sync_m16n8k16_a_f16_2[0]), "r"(_mma_sync_m16n8k16_a_f16_2[1]), "r"(_mma_sync_m16n8k16_a_f16_2[2]), "r"(_mma_sync_m16n8k16_a_f16_2[3]), "r"(_mma_sync_m16n8k16_b_2[0]), "r"(_mma_sync_m16n8k16_b_2[1]));
        int a_group_base_21 = a_base;
        int a_k_byte_22 = (48 + lane / 16 * 8) * 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(((unsigned int)a_group_base_21 + (lane % 16 * 128 + (unsigned int)a_k_byte_22 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_22 >> 7 & 7) << 4)))
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
        int n_region_23 = virtual_warp & 1;
        int u32_pos_24 = (unsigned int)(logical_n_warp * 32) + lane;
        int packed_panel_25 = u32_pos_24 / 32;
        int packed_panel_word_26 = u32_pos_24 - packed_panel_25 * 32;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])) : "r"((b_base + ((6 + packed_panel_25) * 128 + packed_panel_word_26 * 4 ^ ((6 + packed_panel_25) * 128 + packed_panel_word_26 * 4 >> 7 & 7) << 4))));
        int sf_linear_27 = 96 + base_n + logical_n_warp * 8 + warp / 2 * 16;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_27 / 4 * 4));
        uint8_t scale_byte_28 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_27 & 3) * 8) & 255);
        int byte_shift_29 = n_region_23 * 16;
        uint32_t _fp4_dequant_x2_6;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)byte_shift_29 & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_28)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_6) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _fp4_dequant_x2_7;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)(byte_shift_29 + 8) & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_28)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_7) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _mma_sync_m16n8k16_b_3[2];
        _mma_sync_m16n8k16_b_3[0] = _fp4_dequant_x2_6;
        _mma_sync_m16n8k16_b_3[1] = _fp4_dequant_x2_7;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "r"(_mma_sync_m16n8k16_a_f16_3[0]), "r"(_mma_sync_m16n8k16_a_f16_3[1]), "r"(_mma_sync_m16n8k16_a_f16_3[2]), "r"(_mma_sync_m16n8k16_a_f16_3[3]), "r"(_mma_sync_m16n8k16_b_3[0]), "r"(_mma_sync_m16n8k16_b_3[1]));
        int a_group_base_30 = a_base + 2048;
        int a_k_byte_31 = lane / 16 * 8 * 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(((unsigned int)a_group_base_30 + (lane % 16 * 128 + (unsigned int)a_k_byte_31 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_31 >> 7 & 7) << 4)))
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
        int n_region_32 = virtual_warp & 1;
        int u32_pos_33 = (unsigned int)(logical_n_warp * 32) + lane;
        int packed_panel_34 = u32_pos_33 / 32;
        int packed_panel_word_35 = u32_pos_33 - packed_panel_34 * 32;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])) : "r"((b_base + ((8 + packed_panel_34) * 128 + packed_panel_word_35 * 4 ^ ((8 + packed_panel_34) * 128 + packed_panel_word_35 * 4 >> 7 & 7) << 4))));
        int sf_linear_36 = 128 + base_n + logical_n_warp * 8 + warp / 2 * 16;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_36 / 4 * 4));
        uint8_t scale_byte_37 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_36 & 3) * 8) & 255);
        int byte_shift_38 = n_region_32 * 16;
        uint32_t _fp4_dequant_x2_8;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)byte_shift_38 & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_37)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_8) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _fp4_dequant_x2_9;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)(byte_shift_38 + 8) & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_37)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_9) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _mma_sync_m16n8k16_b_4[2];
        _mma_sync_m16n8k16_b_4[0] = _fp4_dequant_x2_8;
        _mma_sync_m16n8k16_b_4[1] = _fp4_dequant_x2_9;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "r"(_mma_sync_m16n8k16_a_f16_4[0]), "r"(_mma_sync_m16n8k16_a_f16_4[1]), "r"(_mma_sync_m16n8k16_a_f16_4[2]), "r"(_mma_sync_m16n8k16_a_f16_4[3]), "r"(_mma_sync_m16n8k16_b_4[0]), "r"(_mma_sync_m16n8k16_b_4[1]));
        int a_group_base_39 = a_base + 2048;
        int a_k_byte_40 = (16 + lane / 16 * 8) * 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(((unsigned int)a_group_base_39 + (lane % 16 * 128 + (unsigned int)a_k_byte_40 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_40 >> 7 & 7) << 4)))
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
        int n_region_41 = virtual_warp & 1;
        int u32_pos_42 = (unsigned int)(logical_n_warp * 32) + lane;
        int packed_panel_43 = u32_pos_42 / 32;
        int packed_panel_word_44 = u32_pos_42 - packed_panel_43 * 32;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])) : "r"((b_base + ((10 + packed_panel_43) * 128 + packed_panel_word_44 * 4 ^ ((10 + packed_panel_43) * 128 + packed_panel_word_44 * 4 >> 7 & 7) << 4))));
        int sf_linear_45 = 160 + base_n + logical_n_warp * 8 + warp / 2 * 16;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_45 / 4 * 4));
        uint8_t scale_byte_46 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_45 & 3) * 8) & 255);
        int byte_shift_47 = n_region_41 * 16;
        uint32_t _fp4_dequant_x2_10;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)byte_shift_47 & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_46)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_10) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _fp4_dequant_x2_11;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)(byte_shift_47 + 8) & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_46)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_11) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _mma_sync_m16n8k16_b_5[2];
        _mma_sync_m16n8k16_b_5[0] = _fp4_dequant_x2_10;
        _mma_sync_m16n8k16_b_5[1] = _fp4_dequant_x2_11;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "r"(_mma_sync_m16n8k16_a_f16_5[0]), "r"(_mma_sync_m16n8k16_a_f16_5[1]), "r"(_mma_sync_m16n8k16_a_f16_5[2]), "r"(_mma_sync_m16n8k16_a_f16_5[3]), "r"(_mma_sync_m16n8k16_b_5[0]), "r"(_mma_sync_m16n8k16_b_5[1]));
        int a_group_base_48 = a_base + 2048;
        int a_k_byte_49 = (32 + lane / 16 * 8) * 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(((unsigned int)a_group_base_48 + (lane % 16 * 128 + (unsigned int)a_k_byte_49 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_49 >> 7 & 7) << 4)))
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
        int n_region_50 = virtual_warp & 1;
        int u32_pos_51 = (unsigned int)(logical_n_warp * 32) + lane;
        int packed_panel_52 = u32_pos_51 / 32;
        int packed_panel_word_53 = u32_pos_51 - packed_panel_52 * 32;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])) : "r"((b_base + ((12 + packed_panel_52) * 128 + packed_panel_word_53 * 4 ^ ((12 + packed_panel_52) * 128 + packed_panel_word_53 * 4 >> 7 & 7) << 4))));
        int sf_linear_54 = 192 + base_n + logical_n_warp * 8 + warp / 2 * 16;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_54 / 4 * 4));
        uint8_t scale_byte_55 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_54 & 3) * 8) & 255);
        int byte_shift_56 = n_region_50 * 16;
        uint32_t _fp4_dequant_x2_12;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)byte_shift_56 & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_55)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_12) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _fp4_dequant_x2_13;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)(byte_shift_56 + 8) & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_55)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_13) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _mma_sync_m16n8k16_b_6[2];
        _mma_sync_m16n8k16_b_6[0] = _fp4_dequant_x2_12;
        _mma_sync_m16n8k16_b_6[1] = _fp4_dequant_x2_13;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "r"(_mma_sync_m16n8k16_a_f16_6[0]), "r"(_mma_sync_m16n8k16_a_f16_6[1]), "r"(_mma_sync_m16n8k16_a_f16_6[2]), "r"(_mma_sync_m16n8k16_a_f16_6[3]), "r"(_mma_sync_m16n8k16_b_6[0]), "r"(_mma_sync_m16n8k16_b_6[1]));
        int a_group_base_57 = a_base + 2048;
        int a_k_byte_58 = (48 + lane / 16 * 8) * 2;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(((unsigned int)a_group_base_57 + (lane % 16 * 128 + (unsigned int)a_k_byte_58 ^ (lane % 16 * 128 + (unsigned int)a_k_byte_58 >> 7 & 7) << 4)))
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
        int n_region_59 = virtual_warp & 1;
        int u32_pos_60 = (unsigned int)(logical_n_warp * 32) + lane;
        int packed_panel_61 = u32_pos_60 / 32;
        int packed_panel_word_62 = u32_pos_60 - packed_panel_61 * 32;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])) : "r"((b_base + ((14 + packed_panel_61) * 128 + packed_panel_word_62 * 4 ^ ((14 + packed_panel_61) * 128 + packed_panel_word_62 * 4 >> 7 & 7) << 4))));
        int sf_linear_63 = 224 + base_n + logical_n_warp * 8 + warp / 2 * 16;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + sf_linear_63 / 4 * 4));
        uint8_t scale_byte_64 = (uint8_t)(scale_word[0] >> (unsigned int)((sf_linear_63 & 3) * 8) & 255);
        int byte_shift_65 = n_region_59 * 16;
        uint32_t _fp4_dequant_x2_14;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)byte_shift_65 & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_64)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_14) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _fp4_dequant_x2_15;
        {
            uint16_t _fp4_u16 = (uint16_t)(((uint32_t)(raw[0] >> (unsigned int)(byte_shift_65 + 8) & 255)) & 0xFFu);
            uint32_t _fp4_x16x2;
            asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
                "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
                "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;         }"
                : "=r"(_fp4_x16x2) : "h"(_fp4_u16));
            uint32_t _scale_byte = ((uint32_t)(scale_byte_64)) & 0xFFu;
            uint32_t _scale_f16x2;
            asm("mul.lo.u32 %0, %1, 0x00800080;" : "=r"(_scale_f16x2) : "r"(_scale_byte));
            uint32_t _scale_x16x2 = _scale_f16x2;
            asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_fp4_dequant_x2_15) : "r"(_fp4_x16x2), "r"(_scale_x16x2));
        }
        uint32_t _mma_sync_m16n8k16_b_7[2];
        _mma_sync_m16n8k16_b_7[0] = _fp4_dequant_x2_14;
        _mma_sync_m16n8k16_b_7[1] = _fp4_dequant_x2_15;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
            : "r"(_mma_sync_m16n8k16_a_f16_7[0]), "r"(_mma_sync_m16n8k16_a_f16_7[1]), "r"(_mma_sync_m16n8k16_a_f16_7[2]), "r"(_mma_sync_m16n8k16_a_f16_7[3]), "r"(_mma_sync_m16n8k16_b_7[0]), "r"(_mma_sync_m16n8k16_b_7[1]));
        __syncthreads();
        int k_tiles_66 = (K + 128 - 1) / 128;
        bool stage_valid_67 = k_tiles_66 > local_kt + 3;
        int _min_33 = ((local_kt + 3) < (k_tiles_66 - 1) ? (local_kt + 3) : (k_tiles_66 - 1));
        int safe_local_kt_68 = _min_33;
        int total_k_groups_69 = K / 16;
        int chunk_70 = tid;
        int local_m_71 = chunk_70 / 16;
        int local_k_72 = chunk_70 % 16 * 8;
        int global_m_73 = off_m + local_m_71;
        int global_k_74 = safe_local_kt_68 * 128 + local_k_72;
        int _min_34 = ((global_m_73) < (M - 1) ? (global_m_73) : (M - 1));
        int safe_m_75 = _min_34;
        int _min_35 = ((global_k_74) < (K - 8) ? (global_k_74) : (K - 8));
        int safe_k_76 = _min_35;
        bool valid_a_77 = stage_valid_67 && global_m_73 < M && global_k_74 < K;
        int a_plane_78 = local_k_72 / 64;
        int a_plane_k_79 = local_k_72 - a_plane_78 * 64;
        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
            :: "r"((cute_remaining_small_a_addr + (unsigned int)(stage * 7168) + (unsigned int)(a_plane_78 * 16 * 64 * 2) + (unsigned int)(local_m_71 * 128 + a_plane_k_79 * 2 ^ (local_m_71 * 128 + a_plane_k_79 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_75 * K + safe_k_76)), "r"((valid_a_77) ? 16 : 0));
        int chunk_80 = tid + 128;
        int local_m_81 = chunk_80 / 16;
        int local_k_82 = chunk_80 % 16 * 8;
        int global_m_83 = off_m + local_m_81;
        int global_k_84 = safe_local_kt_68 * 128 + local_k_82;
        int _min_36 = ((global_m_83) < (M - 1) ? (global_m_83) : (M - 1));
        int safe_m_85 = _min_36;
        int _min_37 = ((global_k_84) < (K - 8) ? (global_k_84) : (K - 8));
        int safe_k_86 = _min_37;
        bool valid_a_87 = stage_valid_67 && global_m_83 < M && global_k_84 < K;
        int a_plane_88 = local_k_82 / 64;
        int a_plane_k_89 = local_k_82 - a_plane_88 * 64;
        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
            :: "r"((cute_remaining_small_a_addr + (unsigned int)(stage * 7168) + (unsigned int)(a_plane_88 * 16 * 64 * 2) + (unsigned int)(local_m_81 * 128 + a_plane_k_89 * 2 ^ (local_m_81 * 128 + a_plane_k_89 * 2 >> 7 & 7) << 4))), "l"(A + (safe_m_85 * K + safe_k_86)), "r"((valid_a_87) ? 16 : 0));
        int chunk_91 = tid;
        int local_k_group_92 = chunk_91 / 64;
        int local_word_93 = chunk_91 % 64;
        int packed_panel_94 = local_word_93 / 32;
        int packed_panel_word_95 = local_word_93 - packed_panel_94 * 32;
        int packed_row_96 = local_k_group_92 * 2 + packed_panel_94;
        int global_k_group_97 = safe_local_kt_68 * 8 + local_k_group_92;
        int _min_38 = ((global_k_group_97) < (total_k_groups_69 - 1) ? (global_k_group_97) : (total_k_groups_69 - 1));
        int safe_k_group_98 = _min_38;
        bool valid_b_99 = stage_valid_67 && global_k_group_97 < total_k_groups_69;
        asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
            :: "r"((cute_remaining_small_b_addr + (unsigned int)(stage * 7168) + (unsigned int)(packed_row_96 * 128 + packed_panel_word_95 * 4 ^ (packed_row_96 * 128 + packed_panel_word_95 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_98 * (N * 2) + packed_off_n * 2 + local_word_93 * 2 + n_half)), "r"((valid_b_99) ? 4 : 0));
        int chunk_100 = tid + 128;
        int local_k_group_101 = chunk_100 / 64;
        int local_word_102 = chunk_100 % 64;
        int packed_panel_103 = local_word_102 / 32;
        int packed_panel_word_104 = local_word_102 - packed_panel_103 * 32;
        int packed_row_105 = local_k_group_101 * 2 + packed_panel_103;
        int global_k_group_106 = safe_local_kt_68 * 8 + local_k_group_101;
        int _min_39 = ((global_k_group_106) < (total_k_groups_69 - 1) ? (global_k_group_106) : (total_k_groups_69 - 1));
        int safe_k_group_107 = _min_39;
        bool valid_b_108 = stage_valid_67 && global_k_group_106 < total_k_groups_69;
        asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
            :: "r"((cute_remaining_small_b_addr + (unsigned int)(stage * 7168) + (unsigned int)(packed_row_105 * 128 + packed_panel_word_104 * 4 ^ (packed_row_105 * 128 + packed_panel_word_104 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_107 * (N * 2) + packed_off_n * 2 + local_word_102 * 2 + n_half)), "r"((valid_b_108) ? 4 : 0));
        int chunk_110 = tid + 256;
        int local_k_group_111 = chunk_110 / 64;
        int local_word_112 = chunk_110 % 64;
        int packed_panel_113 = local_word_112 / 32;
        int packed_panel_word_114 = local_word_112 - packed_panel_113 * 32;
        int packed_row_115 = local_k_group_111 * 2 + packed_panel_113;
        int global_k_group_116 = safe_local_kt_68 * 8 + local_k_group_111;
        int _min_40 = ((global_k_group_116) < (total_k_groups_69 - 1) ? (global_k_group_116) : (total_k_groups_69 - 1));
        int safe_k_group_117 = _min_40;
        bool valid_b_118 = stage_valid_67 && global_k_group_116 < total_k_groups_69;
        asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
            :: "r"((cute_remaining_small_b_addr + (unsigned int)(stage * 7168) + (unsigned int)(packed_row_115 * 128 + packed_panel_word_114 * 4 ^ (packed_row_115 * 128 + packed_panel_word_114 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_117 * (N * 2) + packed_off_n * 2 + local_word_112 * 2 + n_half)), "r"((valid_b_118) ? 4 : 0));
        int chunk_120 = tid + 384;
        int local_k_group_121 = chunk_120 / 64;
        int local_word_122 = chunk_120 % 64;
        int packed_panel_123 = local_word_122 / 32;
        int packed_panel_word_124 = local_word_122 - packed_panel_123 * 32;
        int packed_row_125 = local_k_group_121 * 2 + packed_panel_123;
        int global_k_group_126 = safe_local_kt_68 * 8 + local_k_group_121;
        int _min_41 = ((global_k_group_126) < (total_k_groups_69 - 1) ? (global_k_group_126) : (total_k_groups_69 - 1));
        int safe_k_group_127 = _min_41;
        bool valid_b_128 = stage_valid_67 && global_k_group_126 < total_k_groups_69;
        asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], 4, %2;"
            :: "r"((cute_remaining_small_b_addr + (unsigned int)(stage * 7168) + (unsigned int)(packed_row_125 * 128 + packed_panel_word_124 * 4 ^ (packed_row_125 * 128 + packed_panel_word_124 * 4 >> 7 & 7) << 4))), "l"(B + (safe_k_group_127 * (N * 2) + packed_off_n * 2 + local_word_122 * 2 + n_half)), "r"((valid_b_128) ? 4 : 0));
        unsigned int _min_42 = ((lane) < (3) ? (lane) : (3));
        int scale_chunk_129 = (unsigned int)(warp * 4) + _min_42;
        int local_scale_k_group_130 = scale_chunk_129 / 2;
        int local_scale_n_131 = scale_chunk_129 % 2 * 16;
        int global_scale_k_group_132 = safe_local_kt_68 * 8 + local_scale_k_group_130;
        int _min_43 = ((global_scale_k_group_132) < (total_k_groups_69 - 1) ? (global_scale_k_group_132) : (total_k_groups_69 - 1));
        int safe_scale_k_group_133 = _min_43;
        bool valid_scale_134 = stage_valid_67 && global_scale_k_group_132 < total_k_groups_69;
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %0, 0;\n\t"
            "@p cp.async.cg.shared::cta.global.L2::128B [%1], [%2], 16;\n\t"
            "}"
            :: "r"((valid_scale_134 && lane < 4) ? 1 : 0), "r"(cute_remaining_small_scale_addr + (unsigned int)(stage * 7168) + (unsigned int)(scale_chunk_129 * 16)), "l"(B_descale + (safe_scale_k_group_133 * N + off_n + n_half * 32 + local_scale_n_131)));
        asm volatile("cp.async.commit_group;");
    }
    float alpha_value = 1.0f;
    {
        alpha_value = alpha[0];
    }
    int row0 = lane / 4;
    int row1 = row0 + 8;
    int output_partition = virtual_warp * 16 + logical_n_warp * 8;
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
    }
    if (off_m + row1 < M && col_pair < N) {
        long long row1_base = (long long)(off_m + row1) * (long long)N + (long long)col_pair;
        {
            const float2 _prescale2_1 = {alpha_value, alpha_value};
            #if __CUDA_ARCH__ >= 1000
            #pragma unroll
            for (int _ps = 0; _ps < 1; _ps++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc0[2])[_ps], _prescale2_1);
            #else
            #pragma unroll
            for (int _ps = 0; _ps < 2; _ps++)
                acc0[2 + _ps] *= alpha_value;
            #endif
            __nv_bfloat162 _pk = __floats2bfloat162_rn(acc0[2 + 0], acc0[2 + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + row1_base))[0]) = _pk;
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
