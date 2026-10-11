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
#define SMEM_RANK_BUF_STAGE_BYTES 8192
#define SMEM_RANK_BUF_STRIDE 8192
#define SMEM_LOCAL_BUF_OFF 20736
#define SMEM_LOCAL_BUF_STAGE_BYTES 8192
#define SMEM_LOCAL_BUF_STRIDE 8192
#define SMEM_ROW_BUF_OFF 28928
#define SMEM_ROW_BUF_STAGE_BYTES 2
#define SMEM_ROW_BUF_STRIDE 2
#define SMEM_TOTAL 28928
#define THREADS 1024

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(1024) __cluster_dims__(2,1,1) void
kernel_cake_mxfp4_situ_moe_558b8c131a7f8d0b41e3(int* __restrict__ ids_src, __nv_bfloat16* __restrict__ weights_src, int* __restrict__ ids_dst, float* __restrict__ weights_dst, unsigned int* __restrict__ output, int* __restrict__ tile_expert, int* __restrict__ tile_limit, int* __restrict__ expanded, int* __restrict__ permuted, int* __restrict__ padded_total, int* __restrict__ active_total, int* __restrict__ wide_list, int* __restrict__ wide_count, int* __restrict__ narrow_list, int* __restrict__ narrow_count, int* __restrict__ all_list, int* __restrict__ all_count, int* __restrict__ wide_expert, int* __restrict__ wide_limit, int* __restrict__ chunk_counts, int num_routes, int top_k, int output_words, int num_experts, int local_experts, int local_offset, int tile_size, int narrow_tile, int wide_min_rows, int wide_min_permille, int wide_tile, int ids_stride0, int ids_stride1, int w_stride0, int w_stride1)
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
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

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
    int16_t* local_buf = reinterpret_cast<int16_t*>(smem_raw + 20736);
    const int local_buf_addr = smem + 20736;
    int16_t* row_buf = reinterpret_cast<int16_t*>(smem_raw + 28928);
    const int row_buf_addr = smem + 28928;

    // === Task calls (dependency order) ===
    int thread = tid;
    int block = blockIdx.x;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (block < 2) {
        int lane_0 = lane;
        int warp_1 = warp;
        int chunk = num_routes;
        {
            chunk = (num_routes + 2047) / 2048 * 1024;
        }
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
        {
            chunk_counts[block * 1024 + thread] = count;
            asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            int total = 0;
            int pre_0 = 0;
            int part = chunk_counts[tid];
            total = total + part;
            if (block > 0) {
                pre_0 = pre_0 + part;
            }
            int part_1 = chunk_counts[1024 + tid];
            total = total + part_1;
            if (block > 1) {
                pre_0 = pre_0 + part_1;
            }
            count = total;
            pre = pre_0;
        }
        pre_buf[thread] = pre;
        int wide = 0;
        int nwide = 0;
        int ntiles = 0;
        int within_warp = 0;
        {
            ntiles = (count + tile_size - 1) / tile_size * (1 - wide);
            int acc = ntiles;
            int lane_1 = lane;
            int _shfl_up_20 = __shfl_up_sync(0xFFFFFFFF, acc, 1, 32);
            int other = _shfl_up_20;
            if (lane_1 >= 1) {
                acc = acc + other;
            }
            int _shfl_up_21 = __shfl_up_sync(0xFFFFFFFF, acc, 2, 32);
            int other_2 = _shfl_up_21;
            if (lane_1 >= 2) {
                acc = acc + other_2;
            }
            int _shfl_up_22 = __shfl_up_sync(0xFFFFFFFF, acc, 4, 32);
            int other_3 = _shfl_up_22;
            if (lane_1 >= 4) {
                acc = acc + other_3;
            }
            int _shfl_up_23 = __shfl_up_sync(0xFFFFFFFF, acc, 8, 32);
            int other_4 = _shfl_up_23;
            if (lane_1 >= 8) {
                acc = acc + other_4;
            }
            int _shfl_up_24 = __shfl_up_sync(0xFFFFFFFF, acc, 16, 32);
            int other_5 = _shfl_up_24;
            if (lane_1 >= 16) {
                acc = acc + other_5;
            }
            within_warp = acc;
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
            int acc_1 = subtotal;
            int lane_1_1 = lane;
            int _shfl_up_25 = __shfl_up_sync(0xFFFFFFFF, acc_1, 1, 32);
            int other_1 = _shfl_up_25;
            if (lane_1_1 >= 1) {
                acc_1 = acc_1 + other_1;
            }
            int _shfl_up_26 = __shfl_up_sync(0xFFFFFFFF, acc_1, 2, 32);
            int other_2_1 = _shfl_up_26;
            if (lane_1_1 >= 2) {
                acc_1 = acc_1 + other_2_1;
            }
            int _shfl_up_27 = __shfl_up_sync(0xFFFFFFFF, acc_1, 4, 32);
            int other_3_1 = _shfl_up_27;
            if (lane_1_1 >= 4) {
                acc_1 = acc_1 + other_3_1;
            }
            int _shfl_up_28 = __shfl_up_sync(0xFFFFFFFF, acc_1, 8, 32);
            int other_4_1 = _shfl_up_28;
            if (lane_1_1 >= 8) {
                acc_1 = acc_1 + other_4_1;
            }
            int _shfl_up_29 = __shfl_up_sync(0xFFFFFFFF, acc_1, 16, 32);
            int other_5_1 = _shfl_up_29;
            if (lane_1_1 >= 16) {
                acc_1 = acc_1 + other_5_1;
            }
            subtotal = acc_1;
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
        int clear_block = block - 2;
        if (clear_block >= 0) {
            unsigned int zeros[4];
            zeros[0] = 0;
            zeros[1] = 0;
            zeros[2] = 0;
            zeros[3] = 0;
            int num_vec = output_words / 4;
            int grid_x = gridDim.x;
            int stride = (grid_x - 2) * 1024;
            int first = clear_block * 1024 + thread;
            #pragma unroll 1
            for (int vec = first; vec < num_vec; vec += stride) {
                reinterpret_cast<int4*>(output + (vec * 4))[0] = reinterpret_cast<int4*>(zeros)[0];
            }
            int tail = num_vec * 4 + first;
            if (tail < output_words) {
                output[tail] = 0;
            }
        }
    }
}

} // extern "C"
