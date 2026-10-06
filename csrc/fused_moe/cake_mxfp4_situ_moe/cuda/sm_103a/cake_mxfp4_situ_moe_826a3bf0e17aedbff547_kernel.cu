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
kernel_cake_mxfp4_situ_moe_826a3bf0e17aedbff547(int* __restrict__ topk_ids, int* __restrict__ expert_counts, int* __restrict__ tile_expert, int* __restrict__ tile_limit, int* __restrict__ expanded, int* __restrict__ permuted, int* __restrict__ padded_total, int* __restrict__ active_total, int* __restrict__ alt_expert, int* __restrict__ alt_limit, int* __restrict__ alt_active, int* __restrict__ base_active, int* __restrict__ wide_list, int* __restrict__ wide_count, int* __restrict__ alt_wide_list, int* __restrict__ alt_wide_count, int* __restrict__ narrow_list, int* __restrict__ narrow_count, int* __restrict__ narrow_count_base, int num_tokens, int num_experts, int top_k, int local_offset, int local_experts, int stride_log2, int padding_log2, int padding_log2_alt, int permille, int mixed_narrow_tile, int mixed_row_unit, int mixed_max_rows, int mixed_min_total_rows, int contiguous_windows)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
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
    if (contiguous_windows != 0 && per_block_req <= 3584) {
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
    int expert_regs[4];
    int offset_regs[4];
    int local_extent = local_experts << stride_log2;
    #pragma unroll
    for (int ii0 = 0; ii0 < 4; ii0 += 4) {
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
        int num_cta = count + (1 << padding_log2) - 1 >> padding_log2;
        int lane_0 = lane;
        int warp_2 = warp;
        int lane_3 = lane;
        int acc = num_cta;
        int _shfl_up_24 = __shfl_up_sync(0xFFFFFFFF, acc, 1, 32);
        int up = _shfl_up_24;
        if (lane_3 >= 1) {
            acc = acc + up;
        }
        int _shfl_up_25 = __shfl_up_sync(0xFFFFFFFF, acc, 2, 32);
        int up_4 = _shfl_up_25;
        if (lane_3 >= 2) {
            acc = acc + up_4;
        }
        int _shfl_up_26 = __shfl_up_sync(0xFFFFFFFF, acc, 4, 32);
        int up_5 = _shfl_up_26;
        if (lane_3 >= 4) {
            acc = acc + up_5;
        }
        int _shfl_up_27 = __shfl_up_sync(0xFFFFFFFF, acc, 8, 32);
        int up_6 = _shfl_up_27;
        if (lane_3 >= 8) {
            acc = acc + up_6;
        }
        int _shfl_up_28 = __shfl_up_sync(0xFFFFFFFF, acc, 16, 32);
        int up_7 = _shfl_up_28;
        if (lane_3 >= 16) {
            acc = acc + up_7;
        }
        int inc = acc;
        int _shfl_up_29 = __shfl_up_sync(0xFFFFFFFF, inc, 1, 32);
        int exc = _shfl_up_29;
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
        int local_expert_s = tid_0 - local_offset >> stride_log2;
        #pragma unroll 1
        for (int cta = block; cta < num_cta; cta += nblocks) {
            tile_expert[exc + cta] = local_expert_s;
            int limit1 = exc + cta + 1 << padding_log2;
            int limit2 = (exc << padding_log2) + count;
            int _min_3 = ((limit1) < (limit2) ? (limit1) : (limit2));
            tile_limit[exc + cta] = _min_3;
        }
        int offset_s = exc << padding_log2;
        int permuted_size = total << padding_log2;
        if (block == 0 && warp_1 == 27) {
            if (_elect_one_0) {
                padded_total[0] = permuted_size;
                active_total[0] = total;
            }
        }
        offsets[tid_0] = offset_s + block_offset;
    }
    __syncthreads();
    #pragma unroll
    for (int ii2 = 0; ii2 < 4; ii2++) {
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
