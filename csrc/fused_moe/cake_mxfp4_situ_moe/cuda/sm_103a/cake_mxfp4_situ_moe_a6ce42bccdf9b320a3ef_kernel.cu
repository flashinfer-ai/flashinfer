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
#define SMEM_COUNTS_STAGE_BYTES 4096
#define SMEM_COUNTS_STRIDE 4096
#define SMEM_BASES_OFF 4096
#define SMEM_BASES_STAGE_BYTES 4096
#define SMEM_BASES_STRIDE 4096
#define SMEM_WARP_SUMS_OFF 8192
#define SMEM_WARP_SUMS_STAGE_BYTES 128
#define SMEM_WARP_SUMS_STRIDE 128
#define SMEM_SCAN_BUF_OFF 8320
#define SMEM_SCAN_BUF_STAGE_BYTES 128
#define SMEM_SCAN_BUF_STRIDE 128
#define SMEM_PRE_BUF_OFF 8448
#define SMEM_PRE_BUF_STAGE_BYTES 4096
#define SMEM_PRE_BUF_STRIDE 4096
#define SMEM_RANK_BUF_OFF 12544
#define SMEM_RANK_BUF_STAGE_BYTES 16384
#define SMEM_RANK_BUF_STRIDE 16384
#define SMEM_LOCAL_BUF_OFF 28928
#define SMEM_LOCAL_BUF_STAGE_BYTES 16384
#define SMEM_LOCAL_BUF_STRIDE 16384
#define SMEM_ROW_BUF_OFF 45312
#define SMEM_ROW_BUF_STAGE_BYTES 2
#define SMEM_ROW_BUF_STRIDE 2
#define SMEM_TOTAL 45312
#define THREADS 1024

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_mxfp4_situ_moe_a6ce42bccdf9b320a3ef(int* __restrict__ ids_src, __nv_bfloat16* __restrict__ weights_src, int* __restrict__ ids_dst, float* __restrict__ weights_dst, unsigned int* __restrict__ output, int* __restrict__ tile_expert, int* __restrict__ tile_limit, int* __restrict__ expanded, int* __restrict__ permuted, int* __restrict__ padded_total, int* __restrict__ active_total, int* __restrict__ wide_list, int* __restrict__ wide_count, int* __restrict__ narrow_list, int* __restrict__ narrow_count, int* __restrict__ all_list, int* __restrict__ all_count, int* __restrict__ wide_expert, int* __restrict__ wide_limit, int* __restrict__ chunk_counts, int num_routes, int top_k, int output_words, int num_experts, int local_experts, int local_offset, int tile_size, int narrow_tile, int wide_min_rows, int wide_min_permille, int wide_tile, int ids_stride0, int ids_stride1, int w_stride0, int w_stride1)
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
    int* bases = reinterpret_cast<int*>(smem_raw + 4096);
    const int bases_addr = smem + 4096;
    int* warp_sums = reinterpret_cast<int*>(smem_raw + 8192);
    const int warp_sums_addr = smem + 8192;
    int* scan_buf = reinterpret_cast<int*>(smem_raw + 8320);
    const int scan_buf_addr = smem + 8320;
    int* pre_buf = reinterpret_cast<int*>(smem_raw + 8448);
    const int pre_buf_addr = smem + 8448;
    int16_t* rank_buf = reinterpret_cast<int16_t*>(smem_raw + 12544);
    const int rank_buf_addr = smem + 12544;
    int16_t* local_buf = reinterpret_cast<int16_t*>(smem_raw + 28928);
    const int local_buf_addr = smem + 28928;
    int16_t* row_buf = reinterpret_cast<int16_t*>(smem_raw + 45312);
    const int row_buf_addr = smem + 45312;

    // === Task calls (dependency order) ===
    int thread = tid;
    int block = blockIdx.x;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (block < 1) {
        int lane_0 = lane;
        int warp_1 = warp;
        int chunk = num_routes;
        int r0 = block * chunk;
        int _min_0 = ((r0 + chunk) < (num_routes) ? (r0 + chunk) : (num_routes));
        int r1 = _min_0;
        int my_routes = r1 - r0;
        counts[thread] = 0;
        __syncthreads();
        int last = r1 - 1;
        if (my_routes > 1024) {
            #pragma unroll 1
            for (int base = r0 + thread; base < r1; base += 8192) {
                int experts[8];
                float weights[8];
                int _min_1 = ((base) < (last) ? (base) : (last));
                int ru = _min_1;
                int tok_u = ru / top_k;
                int slot_u = ru % top_k;
                experts[0] = ids_src[tok_u * ids_stride0 + slot_u * ids_stride1];
                {
                    weights[0] = (float)weights_src[tok_u * w_stride0 + slot_u * w_stride1];
                }
                int _min_2 = ((base + 1024) < (last) ? (base + 1024) : (last));
                int ru_0 = _min_2;
                int tok_u_1 = ru_0 / top_k;
                int slot_u_2 = ru_0 % top_k;
                experts[1] = ids_src[tok_u_1 * ids_stride0 + slot_u_2 * ids_stride1];
                {
                    weights[1] = (float)weights_src[tok_u_1 * w_stride0 + slot_u_2 * w_stride1];
                }
                int _min_3 = ((base + 2048) < (last) ? (base + 2048) : (last));
                int ru_3 = _min_3;
                int tok_u_4 = ru_3 / top_k;
                int slot_u_5 = ru_3 % top_k;
                experts[2] = ids_src[tok_u_4 * ids_stride0 + slot_u_5 * ids_stride1];
                {
                    weights[2] = (float)weights_src[tok_u_4 * w_stride0 + slot_u_5 * w_stride1];
                }
                int _min_4 = ((base + 3072) < (last) ? (base + 3072) : (last));
                int ru_6 = _min_4;
                int tok_u_7 = ru_6 / top_k;
                int slot_u_8 = ru_6 % top_k;
                experts[3] = ids_src[tok_u_7 * ids_stride0 + slot_u_8 * ids_stride1];
                {
                    weights[3] = (float)weights_src[tok_u_7 * w_stride0 + slot_u_8 * w_stride1];
                }
                int _min_5 = ((base + 4096) < (last) ? (base + 4096) : (last));
                int ru_9 = _min_5;
                int tok_u_10 = ru_9 / top_k;
                int slot_u_11 = ru_9 % top_k;
                experts[4] = ids_src[tok_u_10 * ids_stride0 + slot_u_11 * ids_stride1];
                {
                    weights[4] = (float)weights_src[tok_u_10 * w_stride0 + slot_u_11 * w_stride1];
                }
                int _min_6 = ((base + 5120) < (last) ? (base + 5120) : (last));
                int ru_12 = _min_6;
                int tok_u_13 = ru_12 / top_k;
                int slot_u_14 = ru_12 % top_k;
                experts[5] = ids_src[tok_u_13 * ids_stride0 + slot_u_14 * ids_stride1];
                {
                    weights[5] = (float)weights_src[tok_u_13 * w_stride0 + slot_u_14 * w_stride1];
                }
                int _min_7 = ((base + 6144) < (last) ? (base + 6144) : (last));
                int ru_15 = _min_7;
                int tok_u_16 = ru_15 / top_k;
                int slot_u_17 = ru_15 % top_k;
                experts[6] = ids_src[tok_u_16 * ids_stride0 + slot_u_17 * ids_stride1];
                {
                    weights[6] = (float)weights_src[tok_u_16 * w_stride0 + slot_u_17 * w_stride1];
                }
                int _min_8 = ((base + 7168) < (last) ? (base + 7168) : (last));
                int ru_18 = _min_8;
                int tok_u_19 = ru_18 / top_k;
                int slot_u_20 = ru_18 % top_k;
                experts[7] = ids_src[tok_u_19 * ids_stride0 + slot_u_20 * ids_stride1];
                {
                    weights[7] = (float)weights_src[tok_u_19 * w_stride0 + slot_u_20 * w_stride1];
                }
                int r = base;
                if (r < r1) {
                    int expert = experts[0];
                    {
                        expert = expert >> 16;
                        ids_dst[r] = expert;
                    }
                    {
                        weights_dst[r] = weights[0];
                    }
                    int local = expert - local_offset;
                    expanded[r] = -1;
                    int rank = -1;
                    if (expert >= 0 && expert < num_experts && local >= 0 && local < local_experts) {
                        uint32_t _shared_atomic_old_0;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_0) : "r"(static_cast<uint32_t>((counts_addr + 4 * (local)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        rank = (int)_shared_atomic_old_0;
                    }
                    rank_buf[r - r0] = (int16_t)rank;
                    local_buf[r - r0] = (int16_t)local;
                }
                int r_21 = base + 1024;
                if (r_21 < r1) {
                    int expert_1 = experts[1];
                    {
                        expert_1 = expert_1 >> 16;
                        ids_dst[r_21] = expert_1;
                    }
                    {
                        weights_dst[r_21] = weights[1];
                    }
                    int local_1 = expert_1 - local_offset;
                    expanded[r_21] = -1;
                    int rank_1 = -1;
                    if (expert_1 >= 0 && expert_1 < num_experts && local_1 >= 0 && local_1 < local_experts) {
                        uint32_t _shared_atomic_old_1;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_1) : "r"(static_cast<uint32_t>((counts_addr + 4 * (local_1)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        rank_1 = (int)_shared_atomic_old_1;
                    }
                    rank_buf[r_21 - r0] = (int16_t)rank_1;
                    local_buf[r_21 - r0] = (int16_t)local_1;
                }
                int r_22 = base + 2048;
                if (r_22 < r1) {
                    int expert_2 = experts[2];
                    {
                        expert_2 = expert_2 >> 16;
                        ids_dst[r_22] = expert_2;
                    }
                    {
                        weights_dst[r_22] = weights[2];
                    }
                    int local_2 = expert_2 - local_offset;
                    expanded[r_22] = -1;
                    int rank_2 = -1;
                    if (expert_2 >= 0 && expert_2 < num_experts && local_2 >= 0 && local_2 < local_experts) {
                        uint32_t _shared_atomic_old_2;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_2) : "r"(static_cast<uint32_t>((counts_addr + 4 * (local_2)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        rank_2 = (int)_shared_atomic_old_2;
                    }
                    rank_buf[r_22 - r0] = (int16_t)rank_2;
                    local_buf[r_22 - r0] = (int16_t)local_2;
                }
                int r_23 = base + 3072;
                if (r_23 < r1) {
                    int expert_3 = experts[3];
                    {
                        expert_3 = expert_3 >> 16;
                        ids_dst[r_23] = expert_3;
                    }
                    {
                        weights_dst[r_23] = weights[3];
                    }
                    int local_3 = expert_3 - local_offset;
                    expanded[r_23] = -1;
                    int rank_3 = -1;
                    if (expert_3 >= 0 && expert_3 < num_experts && local_3 >= 0 && local_3 < local_experts) {
                        uint32_t _shared_atomic_old_3;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_3) : "r"(static_cast<uint32_t>((counts_addr + 4 * (local_3)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        rank_3 = (int)_shared_atomic_old_3;
                    }
                    rank_buf[r_23 - r0] = (int16_t)rank_3;
                    local_buf[r_23 - r0] = (int16_t)local_3;
                }
                int r_24 = base + 4096;
                if (r_24 < r1) {
                    int expert_4 = experts[4];
                    {
                        expert_4 = expert_4 >> 16;
                        ids_dst[r_24] = expert_4;
                    }
                    {
                        weights_dst[r_24] = weights[4];
                    }
                    int local_4 = expert_4 - local_offset;
                    expanded[r_24] = -1;
                    int rank_4 = -1;
                    if (expert_4 >= 0 && expert_4 < num_experts && local_4 >= 0 && local_4 < local_experts) {
                        uint32_t _shared_atomic_old_4;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_4) : "r"(static_cast<uint32_t>((counts_addr + 4 * (local_4)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        rank_4 = (int)_shared_atomic_old_4;
                    }
                    rank_buf[r_24 - r0] = (int16_t)rank_4;
                    local_buf[r_24 - r0] = (int16_t)local_4;
                }
                int r_25 = base + 5120;
                if (r_25 < r1) {
                    int expert_5 = experts[5];
                    {
                        expert_5 = expert_5 >> 16;
                        ids_dst[r_25] = expert_5;
                    }
                    {
                        weights_dst[r_25] = weights[5];
                    }
                    int local_5 = expert_5 - local_offset;
                    expanded[r_25] = -1;
                    int rank_5 = -1;
                    if (expert_5 >= 0 && expert_5 < num_experts && local_5 >= 0 && local_5 < local_experts) {
                        uint32_t _shared_atomic_old_5;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_5) : "r"(static_cast<uint32_t>((counts_addr + 4 * (local_5)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        rank_5 = (int)_shared_atomic_old_5;
                    }
                    rank_buf[r_25 - r0] = (int16_t)rank_5;
                    local_buf[r_25 - r0] = (int16_t)local_5;
                }
                int r_26 = base + 6144;
                if (r_26 < r1) {
                    int expert_6 = experts[6];
                    {
                        expert_6 = expert_6 >> 16;
                        ids_dst[r_26] = expert_6;
                    }
                    {
                        weights_dst[r_26] = weights[6];
                    }
                    int local_6 = expert_6 - local_offset;
                    expanded[r_26] = -1;
                    int rank_6 = -1;
                    if (expert_6 >= 0 && expert_6 < num_experts && local_6 >= 0 && local_6 < local_experts) {
                        uint32_t _shared_atomic_old_6;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_6) : "r"(static_cast<uint32_t>((counts_addr + 4 * (local_6)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        rank_6 = (int)_shared_atomic_old_6;
                    }
                    rank_buf[r_26 - r0] = (int16_t)rank_6;
                    local_buf[r_26 - r0] = (int16_t)local_6;
                }
                int r_27 = base + 7168;
                if (r_27 < r1) {
                    int expert_7 = experts[7];
                    {
                        expert_7 = expert_7 >> 16;
                        ids_dst[r_27] = expert_7;
                    }
                    {
                        weights_dst[r_27] = weights[7];
                    }
                    int local_7 = expert_7 - local_offset;
                    expanded[r_27] = -1;
                    int rank_7 = -1;
                    if (expert_7 >= 0 && expert_7 < num_experts && local_7 >= 0 && local_7 < local_experts) {
                        uint32_t _shared_atomic_old_7;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_7) : "r"(static_cast<uint32_t>((counts_addr + 4 * (local_7)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        rank_7 = (int)_shared_atomic_old_7;
                    }
                    rank_buf[r_27 - r0] = (int16_t)rank_7;
                    local_buf[r_27 - r0] = (int16_t)local_7;
                }
            }
        } else {
            #pragma unroll 1
            for (int r_1 = r0 + thread; r_1 < r1; r_1 += 1024) {
                int tok = r_1 / top_k;
                int slot = r_1 % top_k;
                int expert_8 = ids_src[tok * ids_stride0 + slot * ids_stride1];
                {
                    expert_8 = expert_8 >> 16;
                    ids_dst[r_1] = expert_8;
                }
                {
                    weights_dst[r_1] = (float)weights_src[tok * w_stride0 + slot * w_stride1];
                }
                int local_8 = expert_8 - local_offset;
                expanded[r_1] = -1;
                int rank_8 = -1;
                if (expert_8 >= 0 && expert_8 < num_experts && local_8 >= 0 && local_8 < local_experts) {
                    uint32_t _shared_atomic_old_8;
                    asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_8) : "r"(static_cast<uint32_t>((counts_addr + 4 * (local_8)))), "r"(static_cast<uint32_t>(1)) : "memory");
                    rank_8 = (int)_shared_atomic_old_8;
                }
                rank_buf[r_1 - r0] = (int16_t)rank_8;
                local_buf[r_1 - r0] = (int16_t)local_8;
            }
        }
        __syncthreads();
        int count = counts[thread];
        int pre = 0;
        pre_buf[thread] = pre;
        int wide = 0;
        int nwide = 0;
        int ntiles = 0;
        int within_warp = 0;
        {
            ntiles = (int)(count > 0);
            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, count > 0);
            unsigned int active_mask = _vote_0;
            unsigned int all_lanes = 4294967295;
            unsigned int inclusive_mask = all_lanes >> (unsigned int)(31 - lane_0);
            int _popc_0 = __popc(active_mask & inclusive_mask);
            within_warp = _popc_0;
        }
        if (lane_0 == 31) {
            warp_sums[warp_1] = within_warp;
        }
        __syncthreads();
        if (warp_1 == 0) {
            int subtotal = 0;
            if (lane_0 < 32) {
                subtotal = warp_sums[lane_0];
            }
            int acc = subtotal;
            int lane_1 = lane;
            int _shfl_up_25 = __shfl_up_sync(0xFFFFFFFF, acc, 1, 32);
            int other = _shfl_up_25;
            if (lane_1 >= 1) {
                acc = acc + other;
            }
            int _shfl_up_26 = __shfl_up_sync(0xFFFFFFFF, acc, 2, 32);
            int other_2 = _shfl_up_26;
            if (lane_1 >= 2) {
                acc = acc + other_2;
            }
            int _shfl_up_27 = __shfl_up_sync(0xFFFFFFFF, acc, 4, 32);
            int other_3 = _shfl_up_27;
            if (lane_1 >= 4) {
                acc = acc + other_3;
            }
            int _shfl_up_28 = __shfl_up_sync(0xFFFFFFFF, acc, 8, 32);
            int other_4 = _shfl_up_28;
            if (lane_1 >= 8) {
                acc = acc + other_4;
            }
            int _shfl_up_29 = __shfl_up_sync(0xFFFFFFFF, acc, 16, 32);
            int other_5 = _shfl_up_29;
            if (lane_1 >= 16) {
                acc = acc + other_5;
            }
            subtotal = acc;
            if (lane_0 < 32) {
                warp_sums[lane_0] = subtotal;
            }
        }
        __syncthreads();
        int preceding_warps = 0;
        if (warp_1 > 0) {
            preceding_warps = warp_sums[warp_1 - 1];
        }
        int tile_base = preceding_warps + within_warp - ntiles;
        int row_base = tile_base * tile_size;
        int wide_total = 0;
        int wide_base = 0;
        int wide_row0 = 0;
        bases[thread] = row_base;
        {
            int nonempty = (int)(count > 0);
            int full = ntiles - nonempty;
            int last_rows = count - full * tile_size;
            int sub = tile_size / narrow_tile;
            int last_sub = (last_rows + narrow_tile - 1) / narrow_tile;
            int wide_rows = 0;
            if (tile_size > wide_min_rows) {
                wide_rows = full * tile_size;
            }
            if (last_rows > wide_min_rows) {
                wide_rows = wide_rows + last_rows;
            }
            int lane_1_1 = lane;
            int warp_2 = warp;
            int acc_1 = count;
            int lane_3 = lane;
            int _shfl_up_40 = __shfl_up_sync(0xFFFFFFFF, acc_1, 1, 32);
            int other_1 = _shfl_up_40;
            if (lane_3 >= 1) {
                acc_1 = acc_1 + other_1;
            }
            int _shfl_up_41 = __shfl_up_sync(0xFFFFFFFF, acc_1, 2, 32);
            int other_4_1 = _shfl_up_41;
            if (lane_3 >= 2) {
                acc_1 = acc_1 + other_4_1;
            }
            int _shfl_up_42 = __shfl_up_sync(0xFFFFFFFF, acc_1, 4, 32);
            int other_5_1 = _shfl_up_42;
            if (lane_3 >= 4) {
                acc_1 = acc_1 + other_5_1;
            }
            int _shfl_up_43 = __shfl_up_sync(0xFFFFFFFF, acc_1, 8, 32);
            int other_6 = _shfl_up_43;
            if (lane_3 >= 8) {
                acc_1 = acc_1 + other_6;
            }
            int _shfl_up_44 = __shfl_up_sync(0xFFFFFFFF, acc_1, 16, 32);
            int other_7 = _shfl_up_44;
            if (lane_3 >= 16) {
                acc_1 = acc_1 + other_7;
            }
            int inclusive = acc_1;
            if (lane_1_1 == 31) {
                scan_buf[warp_2] = inclusive;
            }
            __syncthreads();
            if (warp_2 == 0) {
                int subtotal_1 = 0;
                if (lane_1_1 < 32) {
                    subtotal_1 = scan_buf[lane_1_1];
                }
                int acc_0 = subtotal_1;
                int lane_2 = lane;
                int _shfl_up_45 = __shfl_up_sync(0xFFFFFFFF, acc_0, 1, 32);
                int other_3_1 = _shfl_up_45;
                if (lane_2 >= 1) {
                    acc_0 = acc_0 + other_3_1;
                }
                int _shfl_up_46 = __shfl_up_sync(0xFFFFFFFF, acc_0, 2, 32);
                int other_8 = _shfl_up_46;
                if (lane_2 >= 2) {
                    acc_0 = acc_0 + other_8;
                }
                int _shfl_up_47 = __shfl_up_sync(0xFFFFFFFF, acc_0, 4, 32);
                int other_9 = _shfl_up_47;
                if (lane_2 >= 4) {
                    acc_0 = acc_0 + other_9;
                }
                int _shfl_up_48 = __shfl_up_sync(0xFFFFFFFF, acc_0, 8, 32);
                int other_10 = _shfl_up_48;
                if (lane_2 >= 8) {
                    acc_0 = acc_0 + other_10;
                }
                int _shfl_up_49 = __shfl_up_sync(0xFFFFFFFF, acc_0, 16, 32);
                int other_11 = _shfl_up_49;
                if (lane_2 >= 16) {
                    acc_0 = acc_0 + other_11;
                }
                subtotal_1 = acc_0;
                if (lane_1_1 < 32) {
                    scan_buf[lane_1_1] = subtotal_1;
                }
            }
            __syncthreads();
            int preceding = 0;
            if (warp_2 > 0) {
                preceding = scan_buf[warp_2 - 1];
            }
            int total = scan_buf[31];
            int exclusive = preceding + inclusive - count;
            __syncthreads();
            int lane_8 = lane;
            int warp_9 = warp;
            int acc_10 = wide_rows;
            int lane_11 = lane;
            int _shfl_up_50 = __shfl_up_sync(0xFFFFFFFF, acc_10, 1, 32);
            int other_12 = _shfl_up_50;
            if (lane_11 >= 1) {
                acc_10 = acc_10 + other_12;
            }
            int _shfl_up_51 = __shfl_up_sync(0xFFFFFFFF, acc_10, 2, 32);
            int other_13 = _shfl_up_51;
            if (lane_11 >= 2) {
                acc_10 = acc_10 + other_13;
            }
            int _shfl_up_52 = __shfl_up_sync(0xFFFFFFFF, acc_10, 4, 32);
            int other_14 = _shfl_up_52;
            if (lane_11 >= 4) {
                acc_10 = acc_10 + other_14;
            }
            int _shfl_up_53 = __shfl_up_sync(0xFFFFFFFF, acc_10, 8, 32);
            int other_15 = _shfl_up_53;
            if (lane_11 >= 8) {
                acc_10 = acc_10 + other_15;
            }
            int _shfl_up_54 = __shfl_up_sync(0xFFFFFFFF, acc_10, 16, 32);
            int other_16 = _shfl_up_54;
            if (lane_11 >= 16) {
                acc_10 = acc_10 + other_16;
            }
            int inclusive_17 = acc_10;
            if (lane_8 == 31) {
                scan_buf[warp_9] = inclusive_17;
            }
            __syncthreads();
            if (warp_9 == 0) {
                int subtotal_2 = 0;
                if (lane_8 < 32) {
                    subtotal_2 = scan_buf[lane_8];
                }
                int acc_0_1 = subtotal_2;
                int lane_2_1 = lane;
                int _shfl_up_55 = __shfl_up_sync(0xFFFFFFFF, acc_0_1, 1, 32);
                int other_3_2 = _shfl_up_55;
                if (lane_2_1 >= 1) {
                    acc_0_1 = acc_0_1 + other_3_2;
                }
                int _shfl_up_56 = __shfl_up_sync(0xFFFFFFFF, acc_0_1, 2, 32);
                int other_8_1 = _shfl_up_56;
                if (lane_2_1 >= 2) {
                    acc_0_1 = acc_0_1 + other_8_1;
                }
                int _shfl_up_57 = __shfl_up_sync(0xFFFFFFFF, acc_0_1, 4, 32);
                int other_9_1 = _shfl_up_57;
                if (lane_2_1 >= 4) {
                    acc_0_1 = acc_0_1 + other_9_1;
                }
                int _shfl_up_58 = __shfl_up_sync(0xFFFFFFFF, acc_0_1, 8, 32);
                int other_10_1 = _shfl_up_58;
                if (lane_2_1 >= 8) {
                    acc_0_1 = acc_0_1 + other_10_1;
                }
                int _shfl_up_59 = __shfl_up_sync(0xFFFFFFFF, acc_0_1, 16, 32);
                int other_11_1 = _shfl_up_59;
                if (lane_2_1 >= 16) {
                    acc_0_1 = acc_0_1 + other_11_1;
                }
                subtotal_2 = acc_0_1;
                if (lane_8 < 32) {
                    scan_buf[lane_8] = subtotal_2;
                }
            }
            __syncthreads();
            int preceding_18 = 0;
            if (warp_9 > 0) {
                preceding_18 = scan_buf[warp_9 - 1];
            }
            int total_19 = scan_buf[31];
            int exclusive_20 = preceding_18 + inclusive_17 - wide_rows;
            __syncthreads();
            int effective_min = wide_min_rows;
            if (wide_min_permille > 0) {
                if (total_19 * 1000 < total * wide_min_permille) {
                    effective_min = tile_size;
                }
            }
            int full_wide = (int)(effective_min < tile_size);
            int last_wide = (int)(last_rows > effective_min);
            int n_wide = full * full_wide + last_wide;
            int n_all = full * sub + last_sub;
            int n_narrow = full * sub * (1 - full_wide) + last_sub * (1 - last_wide);
            int lane_21 = lane;
            int warp_22 = warp;
            int acc_23 = n_wide;
            int lane_24 = lane;
            int _shfl_up_60 = __shfl_up_sync(0xFFFFFFFF, acc_23, 1, 32);
            int other_25 = _shfl_up_60;
            if (lane_24 >= 1) {
                acc_23 = acc_23 + other_25;
            }
            int _shfl_up_61 = __shfl_up_sync(0xFFFFFFFF, acc_23, 2, 32);
            int other_26 = _shfl_up_61;
            if (lane_24 >= 2) {
                acc_23 = acc_23 + other_26;
            }
            int _shfl_up_62 = __shfl_up_sync(0xFFFFFFFF, acc_23, 4, 32);
            int other_27 = _shfl_up_62;
            if (lane_24 >= 4) {
                acc_23 = acc_23 + other_27;
            }
            int _shfl_up_63 = __shfl_up_sync(0xFFFFFFFF, acc_23, 8, 32);
            int other_28 = _shfl_up_63;
            if (lane_24 >= 8) {
                acc_23 = acc_23 + other_28;
            }
            int _shfl_up_64 = __shfl_up_sync(0xFFFFFFFF, acc_23, 16, 32);
            int other_29 = _shfl_up_64;
            if (lane_24 >= 16) {
                acc_23 = acc_23 + other_29;
            }
            int inclusive_30 = acc_23;
            if (lane_21 == 31) {
                scan_buf[warp_22] = inclusive_30;
            }
            __syncthreads();
            if (warp_22 == 0) {
                int subtotal_3 = 0;
                if (lane_21 < 32) {
                    subtotal_3 = scan_buf[lane_21];
                }
                int acc_0_2 = subtotal_3;
                int lane_2_2 = lane;
                int _shfl_up_65 = __shfl_up_sync(0xFFFFFFFF, acc_0_2, 1, 32);
                int other_3_3 = _shfl_up_65;
                if (lane_2_2 >= 1) {
                    acc_0_2 = acc_0_2 + other_3_3;
                }
                int _shfl_up_66 = __shfl_up_sync(0xFFFFFFFF, acc_0_2, 2, 32);
                int other_8_2 = _shfl_up_66;
                if (lane_2_2 >= 2) {
                    acc_0_2 = acc_0_2 + other_8_2;
                }
                int _shfl_up_67 = __shfl_up_sync(0xFFFFFFFF, acc_0_2, 4, 32);
                int other_9_2 = _shfl_up_67;
                if (lane_2_2 >= 4) {
                    acc_0_2 = acc_0_2 + other_9_2;
                }
                int _shfl_up_68 = __shfl_up_sync(0xFFFFFFFF, acc_0_2, 8, 32);
                int other_10_2 = _shfl_up_68;
                if (lane_2_2 >= 8) {
                    acc_0_2 = acc_0_2 + other_10_2;
                }
                int _shfl_up_69 = __shfl_up_sync(0xFFFFFFFF, acc_0_2, 16, 32);
                int other_11_2 = _shfl_up_69;
                if (lane_2_2 >= 16) {
                    acc_0_2 = acc_0_2 + other_11_2;
                }
                subtotal_3 = acc_0_2;
                if (lane_21 < 32) {
                    scan_buf[lane_21] = subtotal_3;
                }
            }
            __syncthreads();
            int preceding_31 = 0;
            if (warp_22 > 0) {
                preceding_31 = scan_buf[warp_22 - 1];
            }
            int total_32 = scan_buf[31];
            int exclusive_33 = preceding_31 + inclusive_30 - n_wide;
            __syncthreads();
            wide_base = exclusive_33;
            wide_total = total_32;
            int lane_34 = lane;
            int warp_35 = warp;
            int acc_36 = n_narrow;
            int lane_37 = lane;
            int _shfl_up_70 = __shfl_up_sync(0xFFFFFFFF, acc_36, 1, 32);
            int other_38 = _shfl_up_70;
            if (lane_37 >= 1) {
                acc_36 = acc_36 + other_38;
            }
            int _shfl_up_71 = __shfl_up_sync(0xFFFFFFFF, acc_36, 2, 32);
            int other_39 = _shfl_up_71;
            if (lane_37 >= 2) {
                acc_36 = acc_36 + other_39;
            }
            int _shfl_up_72 = __shfl_up_sync(0xFFFFFFFF, acc_36, 4, 32);
            int other_40 = _shfl_up_72;
            if (lane_37 >= 4) {
                acc_36 = acc_36 + other_40;
            }
            int _shfl_up_73 = __shfl_up_sync(0xFFFFFFFF, acc_36, 8, 32);
            int other_41 = _shfl_up_73;
            if (lane_37 >= 8) {
                acc_36 = acc_36 + other_41;
            }
            int _shfl_up_74 = __shfl_up_sync(0xFFFFFFFF, acc_36, 16, 32);
            int other_42 = _shfl_up_74;
            if (lane_37 >= 16) {
                acc_36 = acc_36 + other_42;
            }
            int inclusive_43 = acc_36;
            if (lane_34 == 31) {
                scan_buf[warp_35] = inclusive_43;
            }
            __syncthreads();
            if (warp_35 == 0) {
                int subtotal_4 = 0;
                if (lane_34 < 32) {
                    subtotal_4 = scan_buf[lane_34];
                }
                int acc_0_3 = subtotal_4;
                int lane_2_3 = lane;
                int _shfl_up_75 = __shfl_up_sync(0xFFFFFFFF, acc_0_3, 1, 32);
                int other_3_4 = _shfl_up_75;
                if (lane_2_3 >= 1) {
                    acc_0_3 = acc_0_3 + other_3_4;
                }
                int _shfl_up_76 = __shfl_up_sync(0xFFFFFFFF, acc_0_3, 2, 32);
                int other_8_3 = _shfl_up_76;
                if (lane_2_3 >= 2) {
                    acc_0_3 = acc_0_3 + other_8_3;
                }
                int _shfl_up_77 = __shfl_up_sync(0xFFFFFFFF, acc_0_3, 4, 32);
                int other_9_3 = _shfl_up_77;
                if (lane_2_3 >= 4) {
                    acc_0_3 = acc_0_3 + other_9_3;
                }
                int _shfl_up_78 = __shfl_up_sync(0xFFFFFFFF, acc_0_3, 8, 32);
                int other_10_3 = _shfl_up_78;
                if (lane_2_3 >= 8) {
                    acc_0_3 = acc_0_3 + other_10_3;
                }
                int _shfl_up_79 = __shfl_up_sync(0xFFFFFFFF, acc_0_3, 16, 32);
                int other_11_3 = _shfl_up_79;
                if (lane_2_3 >= 16) {
                    acc_0_3 = acc_0_3 + other_11_3;
                }
                subtotal_4 = acc_0_3;
                if (lane_34 < 32) {
                    scan_buf[lane_34] = subtotal_4;
                }
            }
            __syncthreads();
            int preceding_44 = 0;
            if (warp_35 > 0) {
                preceding_44 = scan_buf[warp_35 - 1];
            }
            int total_45 = scan_buf[31];
            int exclusive_46 = preceding_44 + inclusive_43 - n_narrow;
            __syncthreads();
            int lane_47 = lane;
            int warp_48 = warp;
            int acc_49 = n_all;
            int lane_50 = lane;
            int _shfl_up_80 = __shfl_up_sync(0xFFFFFFFF, acc_49, 1, 32);
            int other_51 = _shfl_up_80;
            if (lane_50 >= 1) {
                acc_49 = acc_49 + other_51;
            }
            int _shfl_up_81 = __shfl_up_sync(0xFFFFFFFF, acc_49, 2, 32);
            int other_52 = _shfl_up_81;
            if (lane_50 >= 2) {
                acc_49 = acc_49 + other_52;
            }
            int _shfl_up_82 = __shfl_up_sync(0xFFFFFFFF, acc_49, 4, 32);
            int other_53 = _shfl_up_82;
            if (lane_50 >= 4) {
                acc_49 = acc_49 + other_53;
            }
            int _shfl_up_83 = __shfl_up_sync(0xFFFFFFFF, acc_49, 8, 32);
            int other_54 = _shfl_up_83;
            if (lane_50 >= 8) {
                acc_49 = acc_49 + other_54;
            }
            int _shfl_up_84 = __shfl_up_sync(0xFFFFFFFF, acc_49, 16, 32);
            int other_55 = _shfl_up_84;
            if (lane_50 >= 16) {
                acc_49 = acc_49 + other_55;
            }
            int inclusive_56 = acc_49;
            if (lane_47 == 31) {
                scan_buf[warp_48] = inclusive_56;
            }
            __syncthreads();
            if (warp_48 == 0) {
                int subtotal_5 = 0;
                if (lane_47 < 32) {
                    subtotal_5 = scan_buf[lane_47];
                }
                int acc_0_4 = subtotal_5;
                int lane_2_4 = lane;
                int _shfl_up_85 = __shfl_up_sync(0xFFFFFFFF, acc_0_4, 1, 32);
                int other_3_5 = _shfl_up_85;
                if (lane_2_4 >= 1) {
                    acc_0_4 = acc_0_4 + other_3_5;
                }
                int _shfl_up_86 = __shfl_up_sync(0xFFFFFFFF, acc_0_4, 2, 32);
                int other_8_4 = _shfl_up_86;
                if (lane_2_4 >= 2) {
                    acc_0_4 = acc_0_4 + other_8_4;
                }
                int _shfl_up_87 = __shfl_up_sync(0xFFFFFFFF, acc_0_4, 4, 32);
                int other_9_4 = _shfl_up_87;
                if (lane_2_4 >= 4) {
                    acc_0_4 = acc_0_4 + other_9_4;
                }
                int _shfl_up_88 = __shfl_up_sync(0xFFFFFFFF, acc_0_4, 8, 32);
                int other_10_4 = _shfl_up_88;
                if (lane_2_4 >= 8) {
                    acc_0_4 = acc_0_4 + other_10_4;
                }
                int _shfl_up_89 = __shfl_up_sync(0xFFFFFFFF, acc_0_4, 16, 32);
                int other_11_4 = _shfl_up_89;
                if (lane_2_4 >= 16) {
                    acc_0_4 = acc_0_4 + other_11_4;
                }
                subtotal_5 = acc_0_4;
                if (lane_47 < 32) {
                    scan_buf[lane_47] = subtotal_5;
                }
            }
            __syncthreads();
            int preceding_57 = 0;
            if (warp_48 > 0) {
                preceding_57 = scan_buf[warp_48 - 1];
            }
            int total_58 = scan_buf[31];
            int exclusive_59 = preceding_57 + inclusive_56 - n_all;
            __syncthreads();
            if (thread < local_experts && block == 0) {
                #pragma unroll 1
                for (int j = 0; j < full; j++) {
                    int tile_j = tile_base + j;
                    if (full_wide == 1) {
                        wide_list[wide_base + j] = tile_j;
                    } else {
                        #pragma unroll 1
                        for (int s = 0; s < sub; s++) {
                            narrow_list[exclusive_46 + j * sub + s] = tile_j * sub + s;
                        }
                    }
                    #pragma unroll 1
                    for (int s2 = 0; s2 < sub; s2++) {
                        all_list[exclusive_59 + j * sub + s2] = tile_j * sub + s2;
                    }
                }
                if (last_rows > 0) {
                    int tile_l = tile_base + full;
                    if (last_wide == 1) {
                        wide_list[wide_base + full * full_wide] = tile_l;
                    } else {
                        int narrow_off = exclusive_46 + full * sub * (1 - full_wide);
                        #pragma unroll 1
                        for (int s3 = 0; s3 < last_sub; s3++) {
                            narrow_list[narrow_off + s3] = tile_l * sub + s3;
                        }
                    }
                    #pragma unroll 1
                    for (int s4 = 0; s4 < last_sub; s4++) {
                        all_list[exclusive_59 + full * sub + s4] = tile_l * sub + s4;
                    }
                }
            }
            if (thread == 1023 && block == 0) {
                wide_count[0] = wide_total;
                narrow_count[0] = total_45;
                all_count[0] = total_58;
            }
        }
        if (thread < local_experts && block == 0) {
            #pragma unroll 1
            for (int jt = 0; jt < ntiles; jt++) {
                int tile_t = tile_base + jt;
                int limit = (tile_t + 1) * tile_size;
                if (limit > row_base + count) {
                    limit = row_base + count;
                }
                tile_expert[tile_t] = thread;
                tile_limit[tile_t] = limit;
            }
        }
        if (thread == 1023 && block == 0) {
            int active = preceding_warps + within_warp;
            active_total[0] = active;
            {
                padded_total[0] = active * tile_size;
            }
        }
        __syncthreads();
        #pragma unroll 1
        for (int rd = r0 + thread; rd < r1; rd += 1024) {
            int rank_d = (int)rank_buf[rd - r0];
            if (rank_d >= 0) {
                int local_d = (int)local_buf[rd - r0];
                int row_d = bases[local_d] + pre_buf[local_d] + rank_d;
                expanded[rd] = row_d;
                permuted[row_d] = rd;
            }
        }
    }
    {
        int word = block * 1024 + thread;
        if (word < output_words) {
            output[word] = 0;
        }
    }
}

} // extern "C"
