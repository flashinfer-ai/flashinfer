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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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
#define SMEM_COUNTS_OFF 0
#define SMEM_COUNTS_STAGE_BYTES 3584
#define SMEM_COUNTS_STRIDE 3584
#define SMEM_OFFSETS_OFF 3584
#define SMEM_OFFSETS_STAGE_BYTES 3584
#define SMEM_OFFSETS_STRIDE 3584
#define SMEM_AGG_OFF 7168
#define SMEM_AGG_STAGE_BYTES 112
#define SMEM_AGG_STRIDE 112
#define SMEM_TOTAL 7296
#define THREADS 896

#include <math_constants.h>
#include <cooperative_groups.h>

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

extern "C" {

__global__ __launch_bounds__(896) void
kernel_cake_mxfp4_situ_moe_53e6cf9ca5c8d19c6f4a(int* __restrict__ topk_ids, int* __restrict__ expert_counts, int* __restrict__ tile_expert, int* __restrict__ tile_limit, int* __restrict__ expanded, int* __restrict__ permuted, int* __restrict__ padded_total, int* __restrict__ active_total, int* __restrict__ alt_expert, int* __restrict__ alt_limit, int* __restrict__ alt_active, int* __restrict__ base_active, int* __restrict__ wide_list, int* __restrict__ wide_count, int* __restrict__ alt_wide_list, int* __restrict__ alt_wide_count, int* __restrict__ narrow_list, int* __restrict__ narrow_count, int* __restrict__ narrow_count_base, int num_tokens, int num_experts, int top_k, int local_offset, int local_experts, int stride_log2, int padding_log2, int padding_log2_alt, int permille, int mixed_narrow_tile, int mixed_row_unit, int mixed_max_rows, int mixed_min_total_rows, int contiguous_windows)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* counts = reinterpret_cast<int*>(smem_raw + 0);
    const int counts_addr = smem + 0;
    int* offsets = reinterpret_cast<int*>(smem_raw + 3584);
    const int offsets_addr = smem + 3584;
    int* agg = reinterpret_cast<int*>(smem_raw + 7168);
    const int agg_addr = smem + 7168;

    // === Task calls (dependency order) ===
    int tid_0 = tid;
    int block = blockIdx.x;
    int nblocks = gridDim.x;
    int gthread = 896 * block + tid_0;
    int gthreads = nblocks * 896;
    int warp_1 = warp;
    int expanded_size = num_tokens * top_k;
    int per_block_req = ((expanded_size + nblocks - 1) / nblocks + 896 - 1) / 896 * 896;
    int contiguous = 0;
    if (contiguous_windows != 0 && per_block_req <= 57344) {
        contiguous = 1;
    }
    int per_block = 0;
    int idx_end = expanded_size;
    if (contiguous != 0) {
        per_block = per_block_req;
        int _min_0 = ((expanded_size) < ((block + 1) * per_block) ? (expanded_size) : ((block + 1) * per_block));
        idx_end = _min_0;
    }
    counts[tid_0] = 0;
    __syncthreads();
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int expert_regs[64];
    int offset_regs[64];
    int local_extent = local_experts << stride_log2;
    #pragma unroll
    for (int ii0 = 0; ii0 < 64; ii0 += 4) {
        int fast = 0;
        if (contiguous == 0 && expanded_size >= (ii0 + 4) * gthreads) {
            fast = 1;
        }
        int stop = 0;
        if (fast != 0) {
            #pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                int ii = ii0 + jj;
                int idx = gthread + ii * gthreads;
                if (contiguous != 0) {
                    idx = block * per_block + ii * 896 + tid_0;
                }
                int idx_0 = idx;
                int expert = topk_ids[idx_0];
                expert_regs[ii] = expert;
                int local = expert - local_offset;
                int mask = (1 << stride_log2) - 1;
                int flag = 0;
                if (local >= 0 && local < local_extent && (local & mask) == 0) {
                    flag = 1;
                }
                int is_local = flag;
                int rank = 0;
                if (is_local != 0) {
                    uint32_t _shared_atomic_old_0;
                    asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_0) : "r"(static_cast<uint32_t>((counts_addr + 4 * (expert)))), "r"(static_cast<uint32_t>(1)) : "memory");
                    rank = (int)_shared_atomic_old_0;
                }
                offset_regs[ii] = rank;
            }
        } else {
            #pragma unroll
            for (int jj_1 = 0; jj_1 < 4; jj_1++) {
                int ii_g = ii0 + jj_1;
                int idx_1 = gthread + ii_g * gthreads;
                if (contiguous != 0) {
                    idx_1 = block * per_block + ii_g * 896 + tid_0;
                }
                int idx_g = idx_1;
                if (idx_g >= idx_end) {
                    stop = 1;
                    break;
                }
                int expert_1 = topk_ids[idx_g];
                expert_regs[ii_g] = expert_1;
                int local_1 = expert_1 - local_offset;
                int mask_1 = (1 << stride_log2) - 1;
                int flag_1 = 0;
                if (local_1 >= 0 && local_1 < local_extent && (local_1 & mask_1) == 0) {
                    flag_1 = 1;
                }
                int is_local_1 = flag_1;
                int rank_1 = 0;
                if (is_local_1 != 0) {
                    uint32_t _shared_atomic_old_1;
                    asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_1) : "r"(static_cast<uint32_t>((counts_addr + 4 * (expert_1)))), "r"(static_cast<uint32_t>(1)) : "memory");
                    rank_1 = (int)_shared_atomic_old_1;
                }
                offset_regs[ii_g] = rank_1;
            }
            if (stop != 0) {
                break;
            }
        }
    }
    __syncthreads();
    int local_count = counts[tid_0];
    int block_offset = 0;
    if (tid_0 < num_experts) {
        int _atomic_old_0 = atomicAdd(&expert_counts[tid_0], local_count);
        block_offset = _atomic_old_0;
    }
    cooperative_groups::this_grid().sync();
    int count = 0;
    if (tid_0 < num_experts) {
        count = expert_counts[tid_0];
    }
    bool _elect_one_0 = elect_sync();
    {
        int pad_base = count + (1 << padding_log2) - 1 >> padding_log2 << padding_log2;
        int pad_alt = count + (1 << padding_log2_alt) - 1 >> padding_log2_alt << padding_log2_alt;
        int lane_0 = lane;
        int warp_2 = warp;
        int lane_3 = lane;
        int acc = pad_base;
        int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, acc, 1, 32);
        int up = _shfl_up_0;
        if (lane_3 >= 1) {
            acc = acc + up;
        }
        int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, acc, 2, 32);
        int up_4 = _shfl_up_1;
        if (lane_3 >= 2) {
            acc = acc + up_4;
        }
        int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, acc, 4, 32);
        int up_5 = _shfl_up_2;
        if (lane_3 >= 4) {
            acc = acc + up_5;
        }
        int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, acc, 8, 32);
        int up_6 = _shfl_up_3;
        if (lane_3 >= 8) {
            acc = acc + up_6;
        }
        int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, acc, 16, 32);
        int up_7 = _shfl_up_4;
        if (lane_3 >= 16) {
            acc = acc + up_7;
        }
        int inc = acc;
        int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, inc, 1, 32);
        int exc = _shfl_up_5;
        if (lane_0 == 31) {
            agg[warp_2] = inc;
        }
        __syncthreads();
        int total = agg[0];
        int prefix = 0;
        if (warp_2 == 1) {
            prefix = total;
        }
        int item = agg[1];
        total = total + item;
        if (warp_2 == 2) {
            prefix = total;
        }
        int item_8 = agg[2];
        total = total + item_8;
        if (warp_2 == 3) {
            prefix = total;
        }
        int item_9 = agg[3];
        total = total + item_9;
        if (warp_2 == 4) {
            prefix = total;
        }
        int item_10 = agg[4];
        total = total + item_10;
        if (warp_2 == 5) {
            prefix = total;
        }
        int item_11 = agg[5];
        total = total + item_11;
        if (warp_2 == 6) {
            prefix = total;
        }
        int item_12 = agg[6];
        total = total + item_12;
        if (warp_2 == 7) {
            prefix = total;
        }
        int item_13 = agg[7];
        total = total + item_13;
        if (warp_2 == 8) {
            prefix = total;
        }
        int item_14 = agg[8];
        total = total + item_14;
        if (warp_2 == 9) {
            prefix = total;
        }
        int item_15 = agg[9];
        total = total + item_15;
        if (warp_2 == 10) {
            prefix = total;
        }
        int item_16 = agg[10];
        total = total + item_16;
        if (warp_2 == 11) {
            prefix = total;
        }
        int item_17 = agg[11];
        total = total + item_17;
        if (warp_2 == 12) {
            prefix = total;
        }
        int item_18 = agg[12];
        total = total + item_18;
        if (warp_2 == 13) {
            prefix = total;
        }
        int item_19 = agg[13];
        total = total + item_19;
        if (warp_2 == 14) {
            prefix = total;
        }
        int item_20 = agg[14];
        total = total + item_20;
        if (warp_2 == 15) {
            prefix = total;
        }
        int item_21 = agg[15];
        total = total + item_21;
        if (warp_2 == 16) {
            prefix = total;
        }
        int item_22 = agg[16];
        total = total + item_22;
        if (warp_2 == 17) {
            prefix = total;
        }
        int item_23 = agg[17];
        total = total + item_23;
        if (warp_2 == 18) {
            prefix = total;
        }
        int item_24 = agg[18];
        total = total + item_24;
        if (warp_2 == 19) {
            prefix = total;
        }
        int item_25 = agg[19];
        total = total + item_25;
        if (warp_2 == 20) {
            prefix = total;
        }
        int item_26 = agg[20];
        total = total + item_26;
        if (warp_2 == 21) {
            prefix = total;
        }
        int item_27 = agg[21];
        total = total + item_27;
        if (warp_2 == 22) {
            prefix = total;
        }
        int item_28 = agg[22];
        total = total + item_28;
        if (warp_2 == 23) {
            prefix = total;
        }
        int item_29 = agg[23];
        total = total + item_29;
        if (warp_2 == 24) {
            prefix = total;
        }
        int item_30 = agg[24];
        total = total + item_30;
        if (warp_2 == 25) {
            prefix = total;
        }
        int item_31 = agg[25];
        total = total + item_31;
        if (warp_2 == 26) {
            prefix = total;
        }
        int item_32 = agg[26];
        total = total + item_32;
        if (warp_2 == 27) {
            prefix = total;
        }
        int item_33 = agg[27];
        total = total + item_33;
        exc = prefix + exc;
        if (lane_0 == 0) {
            exc = prefix;
        }
        __syncthreads();
        int lane_34 = lane;
        int warp_35 = warp;
        int lane_36 = lane;
        int acc_37 = pad_alt;
        int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, acc_37, 1, 32);
        int up_38 = _shfl_up_6;
        if (lane_36 >= 1) {
            acc_37 = acc_37 + up_38;
        }
        int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, acc_37, 2, 32);
        int up_39 = _shfl_up_7;
        if (lane_36 >= 2) {
            acc_37 = acc_37 + up_39;
        }
        int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, acc_37, 4, 32);
        int up_40 = _shfl_up_8;
        if (lane_36 >= 4) {
            acc_37 = acc_37 + up_40;
        }
        int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, acc_37, 8, 32);
        int up_41 = _shfl_up_9;
        if (lane_36 >= 8) {
            acc_37 = acc_37 + up_41;
        }
        int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, acc_37, 16, 32);
        int up_42 = _shfl_up_10;
        if (lane_36 >= 16) {
            acc_37 = acc_37 + up_42;
        }
        int inc_43 = acc_37;
        int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, inc_43, 1, 32);
        int exc_44 = _shfl_up_11;
        if (lane_34 == 31) {
            agg[warp_35] = inc_43;
        }
        __syncthreads();
        int total_45 = agg[0];
        int prefix_46 = 0;
        if (warp_35 == 1) {
            prefix_46 = total_45;
        }
        int item_47 = agg[1];
        total_45 = total_45 + item_47;
        if (warp_35 == 2) {
            prefix_46 = total_45;
        }
        int item_48 = agg[2];
        total_45 = total_45 + item_48;
        if (warp_35 == 3) {
            prefix_46 = total_45;
        }
        int item_49 = agg[3];
        total_45 = total_45 + item_49;
        if (warp_35 == 4) {
            prefix_46 = total_45;
        }
        int item_50 = agg[4];
        total_45 = total_45 + item_50;
        if (warp_35 == 5) {
            prefix_46 = total_45;
        }
        int item_51 = agg[5];
        total_45 = total_45 + item_51;
        if (warp_35 == 6) {
            prefix_46 = total_45;
        }
        int item_52 = agg[6];
        total_45 = total_45 + item_52;
        if (warp_35 == 7) {
            prefix_46 = total_45;
        }
        int item_53 = agg[7];
        total_45 = total_45 + item_53;
        if (warp_35 == 8) {
            prefix_46 = total_45;
        }
        int item_54 = agg[8];
        total_45 = total_45 + item_54;
        if (warp_35 == 9) {
            prefix_46 = total_45;
        }
        int item_55 = agg[9];
        total_45 = total_45 + item_55;
        if (warp_35 == 10) {
            prefix_46 = total_45;
        }
        int item_56 = agg[10];
        total_45 = total_45 + item_56;
        if (warp_35 == 11) {
            prefix_46 = total_45;
        }
        int item_57 = agg[11];
        total_45 = total_45 + item_57;
        if (warp_35 == 12) {
            prefix_46 = total_45;
        }
        int item_58 = agg[12];
        total_45 = total_45 + item_58;
        if (warp_35 == 13) {
            prefix_46 = total_45;
        }
        int item_59 = agg[13];
        total_45 = total_45 + item_59;
        if (warp_35 == 14) {
            prefix_46 = total_45;
        }
        int item_60 = agg[14];
        total_45 = total_45 + item_60;
        if (warp_35 == 15) {
            prefix_46 = total_45;
        }
        int item_61 = agg[15];
        total_45 = total_45 + item_61;
        if (warp_35 == 16) {
            prefix_46 = total_45;
        }
        int item_62 = agg[16];
        total_45 = total_45 + item_62;
        if (warp_35 == 17) {
            prefix_46 = total_45;
        }
        int item_63 = agg[17];
        total_45 = total_45 + item_63;
        if (warp_35 == 18) {
            prefix_46 = total_45;
        }
        int item_64 = agg[18];
        total_45 = total_45 + item_64;
        if (warp_35 == 19) {
            prefix_46 = total_45;
        }
        int item_65 = agg[19];
        total_45 = total_45 + item_65;
        if (warp_35 == 20) {
            prefix_46 = total_45;
        }
        int item_66 = agg[20];
        total_45 = total_45 + item_66;
        if (warp_35 == 21) {
            prefix_46 = total_45;
        }
        int item_67 = agg[21];
        total_45 = total_45 + item_67;
        if (warp_35 == 22) {
            prefix_46 = total_45;
        }
        int item_68 = agg[22];
        total_45 = total_45 + item_68;
        if (warp_35 == 23) {
            prefix_46 = total_45;
        }
        int item_69 = agg[23];
        total_45 = total_45 + item_69;
        if (warp_35 == 24) {
            prefix_46 = total_45;
        }
        int item_70 = agg[24];
        total_45 = total_45 + item_70;
        if (warp_35 == 25) {
            prefix_46 = total_45;
        }
        int item_71 = agg[25];
        total_45 = total_45 + item_71;
        if (warp_35 == 26) {
            prefix_46 = total_45;
        }
        int item_72 = agg[26];
        total_45 = total_45 + item_72;
        if (warp_35 == 27) {
            prefix_46 = total_45;
        }
        int item_73 = agg[27];
        total_45 = total_45 + item_73;
        exc_44 = prefix_46 + exc_44;
        if (lane_34 == 0) {
            exc_44 = prefix_46;
        }
        __syncthreads();
        int use_alt = 0;
        if ((long long)total_45 * 1000 <= (long long)total * (long long)permille) {
            use_alt = 1;
        }
        int log2_chosen = padding_log2;
        int padded = pad_base;
        if (use_alt != 0) {
            log2_chosen = padding_log2_alt;
            padded = pad_alt;
        }
        int lane_74 = lane;
        int warp_75 = warp;
        int lane_76 = lane;
        int acc_77 = padded;
        int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, acc_77, 1, 32);
        int up_78 = _shfl_up_12;
        if (lane_76 >= 1) {
            acc_77 = acc_77 + up_78;
        }
        int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, acc_77, 2, 32);
        int up_79 = _shfl_up_13;
        if (lane_76 >= 2) {
            acc_77 = acc_77 + up_79;
        }
        int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, acc_77, 4, 32);
        int up_80 = _shfl_up_14;
        if (lane_76 >= 4) {
            acc_77 = acc_77 + up_80;
        }
        int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, acc_77, 8, 32);
        int up_81 = _shfl_up_15;
        if (lane_76 >= 8) {
            acc_77 = acc_77 + up_81;
        }
        int _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, acc_77, 16, 32);
        int up_82 = _shfl_up_16;
        if (lane_76 >= 16) {
            acc_77 = acc_77 + up_82;
        }
        int inc_83 = acc_77;
        int _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, inc_83, 1, 32);
        int exc_84 = _shfl_up_17;
        if (lane_74 == 31) {
            agg[warp_75] = inc_83;
        }
        __syncthreads();
        int total_85 = agg[0];
        int prefix_86 = 0;
        if (warp_75 == 1) {
            prefix_86 = total_85;
        }
        int item_87 = agg[1];
        total_85 = total_85 + item_87;
        if (warp_75 == 2) {
            prefix_86 = total_85;
        }
        int item_88 = agg[2];
        total_85 = total_85 + item_88;
        if (warp_75 == 3) {
            prefix_86 = total_85;
        }
        int item_89 = agg[3];
        total_85 = total_85 + item_89;
        if (warp_75 == 4) {
            prefix_86 = total_85;
        }
        int item_90 = agg[4];
        total_85 = total_85 + item_90;
        if (warp_75 == 5) {
            prefix_86 = total_85;
        }
        int item_91 = agg[5];
        total_85 = total_85 + item_91;
        if (warp_75 == 6) {
            prefix_86 = total_85;
        }
        int item_92 = agg[6];
        total_85 = total_85 + item_92;
        if (warp_75 == 7) {
            prefix_86 = total_85;
        }
        int item_93 = agg[7];
        total_85 = total_85 + item_93;
        if (warp_75 == 8) {
            prefix_86 = total_85;
        }
        int item_94 = agg[8];
        total_85 = total_85 + item_94;
        if (warp_75 == 9) {
            prefix_86 = total_85;
        }
        int item_95 = agg[9];
        total_85 = total_85 + item_95;
        if (warp_75 == 10) {
            prefix_86 = total_85;
        }
        int item_96 = agg[10];
        total_85 = total_85 + item_96;
        if (warp_75 == 11) {
            prefix_86 = total_85;
        }
        int item_97 = agg[11];
        total_85 = total_85 + item_97;
        if (warp_75 == 12) {
            prefix_86 = total_85;
        }
        int item_98 = agg[12];
        total_85 = total_85 + item_98;
        if (warp_75 == 13) {
            prefix_86 = total_85;
        }
        int item_99 = agg[13];
        total_85 = total_85 + item_99;
        if (warp_75 == 14) {
            prefix_86 = total_85;
        }
        int item_100 = agg[14];
        total_85 = total_85 + item_100;
        if (warp_75 == 15) {
            prefix_86 = total_85;
        }
        int item_101 = agg[15];
        total_85 = total_85 + item_101;
        if (warp_75 == 16) {
            prefix_86 = total_85;
        }
        int item_102 = agg[16];
        total_85 = total_85 + item_102;
        if (warp_75 == 17) {
            prefix_86 = total_85;
        }
        int item_103 = agg[17];
        total_85 = total_85 + item_103;
        if (warp_75 == 18) {
            prefix_86 = total_85;
        }
        int item_104 = agg[18];
        total_85 = total_85 + item_104;
        if (warp_75 == 19) {
            prefix_86 = total_85;
        }
        int item_105 = agg[19];
        total_85 = total_85 + item_105;
        if (warp_75 == 20) {
            prefix_86 = total_85;
        }
        int item_106 = agg[20];
        total_85 = total_85 + item_106;
        if (warp_75 == 21) {
            prefix_86 = total_85;
        }
        int item_107 = agg[21];
        total_85 = total_85 + item_107;
        if (warp_75 == 22) {
            prefix_86 = total_85;
        }
        int item_108 = agg[22];
        total_85 = total_85 + item_108;
        if (warp_75 == 23) {
            prefix_86 = total_85;
        }
        int item_109 = agg[23];
        total_85 = total_85 + item_109;
        if (warp_75 == 24) {
            prefix_86 = total_85;
        }
        int item_110 = agg[24];
        total_85 = total_85 + item_110;
        if (warp_75 == 25) {
            prefix_86 = total_85;
        }
        int item_111 = agg[25];
        total_85 = total_85 + item_111;
        if (warp_75 == 26) {
            prefix_86 = total_85;
        }
        int item_112 = agg[26];
        total_85 = total_85 + item_112;
        if (warp_75 == 27) {
            prefix_86 = total_85;
        }
        int item_113 = agg[27];
        total_85 = total_85 + item_113;
        exc_84 = prefix_86 + exc_84;
        if (lane_74 == 0) {
            exc_84 = prefix_86;
        }
        __syncthreads();
        int n_wide = 0;
        int n_narrow = 0;
        int wide_offset = 0;
        int narrow_offset = 0;
        int wide_total = 0;
        int narrow_total = 0;
        int log2_unit = 0;
        {
            int _ffs_0 = __ffs(mixed_row_unit);
            log2_unit = _ffs_0 - 1;
            int rows = 1 << log2_chosen;
            int gu = rows >> log2_unit;
            int windows = 0;
            if (total_85 >= mixed_min_total_rows) {
                windows = 1;
            }
            if (count > 0) {
                if (windows == 0 || mixed_max_rows > 0 && count > mixed_max_rows) {
                    n_wide = padded >> log2_chosen;
                } else {
                    int best_cover = 2147483647;
                    #pragma unroll 1
                    for (int a = 0; a < gu; a++) {
                        int rem = count - a * mixed_narrow_tile;
                        int w_tiles = 0;
                        if (rem > 0) {
                            w_tiles = rem + rows - 1 >> log2_chosen;
                        }
                        int cover = (w_tiles << log2_chosen) + a * mixed_narrow_tile;
                        if (cover < best_cover) {
                            best_cover = cover;
                            n_wide = w_tiles;
                            n_narrow = a;
                        }
                        if (rem <= 0) {
                            break;
                        }
                    }
                }
            }
            int packed = n_wide << 12 | n_narrow;
            int lane_1 = lane;
            int warp_3 = warp;
            int lane_4 = lane;
            int acc_5 = packed;
            int _shfl_up_18 = __shfl_up_sync(0xFFFFFFFF, acc_5, 1, 32);
            int up_8 = _shfl_up_18;
            if (lane_4 >= 1) {
                acc_5 = acc_5 + up_8;
            }
            int _shfl_up_19 = __shfl_up_sync(0xFFFFFFFF, acc_5, 2, 32);
            int up_9 = _shfl_up_19;
            if (lane_4 >= 2) {
                acc_5 = acc_5 + up_9;
            }
            int _shfl_up_20 = __shfl_up_sync(0xFFFFFFFF, acc_5, 4, 32);
            int up_10 = _shfl_up_20;
            if (lane_4 >= 4) {
                acc_5 = acc_5 + up_10;
            }
            int _shfl_up_21 = __shfl_up_sync(0xFFFFFFFF, acc_5, 8, 32);
            int up_11 = _shfl_up_21;
            if (lane_4 >= 8) {
                acc_5 = acc_5 + up_11;
            }
            int _shfl_up_22 = __shfl_up_sync(0xFFFFFFFF, acc_5, 16, 32);
            int up_12 = _shfl_up_22;
            if (lane_4 >= 16) {
                acc_5 = acc_5 + up_12;
            }
            int inc_13 = acc_5;
            int _shfl_up_23 = __shfl_up_sync(0xFFFFFFFF, inc_13, 1, 32);
            int exc_14 = _shfl_up_23;
            if (lane_1 == 31) {
                agg[warp_3] = inc_13;
            }
            __syncthreads();
            int total_15 = agg[0];
            int prefix_16 = 0;
            if (warp_3 == 1) {
                prefix_16 = total_15;
            }
            int item_34 = agg[1];
            total_15 = total_15 + item_34;
            if (warp_3 == 2) {
                prefix_16 = total_15;
            }
            int item_35 = agg[2];
            total_15 = total_15 + item_35;
            if (warp_3 == 3) {
                prefix_16 = total_15;
            }
            int item_36 = agg[3];
            total_15 = total_15 + item_36;
            if (warp_3 == 4) {
                prefix_16 = total_15;
            }
            int item_37 = agg[4];
            total_15 = total_15 + item_37;
            if (warp_3 == 5) {
                prefix_16 = total_15;
            }
            int item_38 = agg[5];
            total_15 = total_15 + item_38;
            if (warp_3 == 6) {
                prefix_16 = total_15;
            }
            int item_39 = agg[6];
            total_15 = total_15 + item_39;
            if (warp_3 == 7) {
                prefix_16 = total_15;
            }
            int item_40 = agg[7];
            total_15 = total_15 + item_40;
            if (warp_3 == 8) {
                prefix_16 = total_15;
            }
            int item_41 = agg[8];
            total_15 = total_15 + item_41;
            if (warp_3 == 9) {
                prefix_16 = total_15;
            }
            int item_42 = agg[9];
            total_15 = total_15 + item_42;
            if (warp_3 == 10) {
                prefix_16 = total_15;
            }
            int item_43 = agg[10];
            total_15 = total_15 + item_43;
            if (warp_3 == 11) {
                prefix_16 = total_15;
            }
            int item_44 = agg[11];
            total_15 = total_15 + item_44;
            if (warp_3 == 12) {
                prefix_16 = total_15;
            }
            int item_45 = agg[12];
            total_15 = total_15 + item_45;
            if (warp_3 == 13) {
                prefix_16 = total_15;
            }
            int item_46 = agg[13];
            total_15 = total_15 + item_46;
            if (warp_3 == 14) {
                prefix_16 = total_15;
            }
            int item_74 = agg[14];
            total_15 = total_15 + item_74;
            if (warp_3 == 15) {
                prefix_16 = total_15;
            }
            int item_75 = agg[15];
            total_15 = total_15 + item_75;
            if (warp_3 == 16) {
                prefix_16 = total_15;
            }
            int item_76 = agg[16];
            total_15 = total_15 + item_76;
            if (warp_3 == 17) {
                prefix_16 = total_15;
            }
            int item_77 = agg[17];
            total_15 = total_15 + item_77;
            if (warp_3 == 18) {
                prefix_16 = total_15;
            }
            int item_78 = agg[18];
            total_15 = total_15 + item_78;
            if (warp_3 == 19) {
                prefix_16 = total_15;
            }
            int item_79 = agg[19];
            total_15 = total_15 + item_79;
            if (warp_3 == 20) {
                prefix_16 = total_15;
            }
            int item_80 = agg[20];
            total_15 = total_15 + item_80;
            if (warp_3 == 21) {
                prefix_16 = total_15;
            }
            int item_81 = agg[21];
            total_15 = total_15 + item_81;
            if (warp_3 == 22) {
                prefix_16 = total_15;
            }
            int item_82 = agg[22];
            total_15 = total_15 + item_82;
            if (warp_3 == 23) {
                prefix_16 = total_15;
            }
            int item_83 = agg[23];
            total_15 = total_15 + item_83;
            if (warp_3 == 24) {
                prefix_16 = total_15;
            }
            int item_84 = agg[24];
            total_15 = total_15 + item_84;
            if (warp_3 == 25) {
                prefix_16 = total_15;
            }
            int item_85 = agg[25];
            total_15 = total_15 + item_85;
            if (warp_3 == 26) {
                prefix_16 = total_15;
            }
            int item_86 = agg[26];
            total_15 = total_15 + item_86;
            if (warp_3 == 27) {
                prefix_16 = total_15;
            }
            int item_114 = agg[27];
            total_15 = total_15 + item_114;
            exc_14 = prefix_16 + exc_14;
            if (lane_1 == 0) {
                exc_14 = prefix_16;
            }
            __syncthreads();
            wide_offset = exc_14 >> 12;
            narrow_offset = exc_14 & 4095;
            wide_total = total_15 >> 12;
            narrow_total = total_15 & 4095;
        }
        if (count > 0) {
            int local_expert = tid_0 - local_offset >> stride_log2;
            int row_limit = exc_84 + count;
            int base_tiles = padded >> padding_log2;
            int base_first = exc_84 >> padding_log2;
            #pragma unroll 1
            for (int t = block; t < base_tiles; t += nblocks) {
                tile_expert[base_first + t] = local_expert;
                int _min_1 = ((exc_84 + (t + 1 << padding_log2)) < (row_limit) ? (exc_84 + (t + 1 << padding_log2)) : (row_limit));
                tile_limit[base_first + t] = _min_1;
            }
            if (use_alt != 0) {
                int alt_tiles = padded >> padding_log2_alt;
                int alt_first = exc_84 >> padding_log2_alt;
                #pragma unroll 1
                for (int t2 = block; t2 < alt_tiles; t2 += nblocks) {
                    alt_expert[alt_first + t2] = local_expert;
                    int _min_2 = ((exc_84 + (t2 + 1 << padding_log2_alt)) < (row_limit) ? (exc_84 + (t2 + 1 << padding_log2_alt)) : (row_limit));
                    alt_limit[alt_first + t2] = _min_2;
                }
            }
            {
                int first = exc_84 >> log2_chosen;
                if (use_alt != 0) {
                    #pragma unroll 1
                    for (int t3 = block; t3 < n_wide; t3 += nblocks) {
                        alt_wide_list[wide_offset + t3] = first + t3;
                    }
                } else {
                    #pragma unroll 1
                    for (int t4 = block; t4 < n_wide; t4 += nblocks) {
                        wide_list[wide_offset + t4] = first + t4;
                    }
                }
                int narrow_row0 = exc_84 + (n_wide << log2_chosen);
                #pragma unroll 1
                for (int i = block; i < n_narrow; i += nblocks) {
                    narrow_list[narrow_offset + i] = narrow_row0 + i * mixed_narrow_tile >> log2_unit;
                }
            }
        }
        if (block == 0 && warp_1 == 27) {
            if (_elect_one_0) {
                {
                    wide_count[0] = 0;
                    alt_wide_count[0] = wide_total;
                    if (use_alt == 0) {
                        wide_count[0] = wide_total;
                        alt_wide_count[0] = 0;
                    }
                    narrow_count[0] = narrow_total;
                    {
                        narrow_count_base[0] = 0;
                        if (use_alt == 0) {
                            narrow_count_base[0] = narrow_total;
                        }
                    }
                }
                int base_count = total_85 >> padding_log2;
                padded_total[0] = total_85;
                active_total[0] = base_count;
                base_active[0] = base_count;
                alt_active[0] = 0;
                if (use_alt != 0) {
                    base_active[0] = 0;
                    alt_active[0] = total_85 >> padding_log2_alt;
                }
            }
        }
        offsets[tid_0] = exc_84 + block_offset;
    }
    __syncthreads();
    #pragma unroll
    for (int ii2 = 0; ii2 < 64; ii2++) {
        int idx_2 = gthread + ii2 * gthreads;
        if (contiguous != 0) {
            idx_2 = block * per_block + ii2 * 896 + tid_0;
        }
        int idx2 = idx_2;
        if (idx2 >= idx_end) {
            break;
        }
        int expert2 = expert_regs[ii2];
        int local_2 = expert2 - local_offset;
        int mask_2 = (1 << stride_log2) - 1;
        int flag_2 = 0;
        if (local_2 >= 0 && local_2 < local_extent && (local_2 & mask_2) == 0) {
            flag_2 = 1;
        }
        int is_local2 = flag_2;
        int permuted_idx = -1;
        if (is_local2 != 0) {
            permuted_idx = offsets[expert2] + offset_regs[ii2];
        }
        expanded[idx2] = permuted_idx;
        if (is_local2 != 0) {
            permuted[permuted_idx] = idx2;
        }
    }
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

} // extern "C"
