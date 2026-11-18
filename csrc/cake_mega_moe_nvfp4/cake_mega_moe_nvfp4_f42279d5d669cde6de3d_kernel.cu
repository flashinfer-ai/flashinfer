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
template <typename T, int Capacity = 8>
struct __align__(16) CakePeerPointerTable { T* ptrs[Capacity]; };
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

#define LAUNCH_MIN_BLOCKS 1

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







__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_mega_moe_nvfp4_f42279d5d669cde6de3d(int* __restrict__ topk_indices, float* __restrict__ topk_scores, long long* sorted_metadata, float* sorted_scores, int* pool_expert_base, int* expert_counts, int* source_expert_base, int* push_destination_base, int* sorted_metadata_ready, int* push_table_ready, int pool_capacity, int mega_pg_world, int mega_pg_rank, const __grid_constant__ CakePeerPointerTable<unsigned int> mega_pg_flags, int* sizes_by_rank, const __grid_constant__ CakePeerPointerTable<int> sizes_by_rank_peers, int* x_ready, const __grid_constant__ CakePeerPointerTable<int> x_ready_peers, int* sizes, const __grid_constant__ CakePeerPointerTable<int> sizes_peers, int* sizes_ready, const __grid_constant__ CakePeerPointerTable<int> sizes_ready_peers, int* metadata_ready, const __grid_constant__ CakePeerPointerTable<int> metadata_ready_peers, long long* token_src_metadata, const __grid_constant__ CakePeerPointerTable<long long> token_src_metadata_peers, float* fc1_topk_scores, const __grid_constant__ CakePeerPointerTable<float> fc1_topk_scores_peers)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(sorted_metadata));
    __builtin_assume(__isGlobal(sorted_metadata));
    asm volatile("" : "+l"(sorted_scores));
    __builtin_assume(__isGlobal(sorted_scores));
    asm volatile("" : "+l"(pool_expert_base));
    __builtin_assume(__isGlobal(pool_expert_base));
    asm volatile("" : "+l"(expert_counts));
    __builtin_assume(__isGlobal(expert_counts));
    asm volatile("" : "+l"(source_expert_base));
    __builtin_assume(__isGlobal(source_expert_base));
    asm volatile("" : "+l"(push_destination_base));
    __builtin_assume(__isGlobal(push_destination_base));

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
    #define matrix_full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    int* histogram = reinterpret_cast<int*>(smem_raw + 1024);
    const int histogram_addr = smem + 1024;
    int* run_start = reinterpret_cast<int*>(smem_raw + 1232);
    const int run_start_addr = smem + 1232;
    int* warp_totals = reinterpret_cast<int*>(smem_raw + 1440);
    const int warp_totals_addr = smem + 1440;
    unsigned long long* sorted_routes = reinterpret_cast<unsigned long long*>(smem_raw + 1504);
    const int sorted_routes_addr = smem + 1504;
    int* size_matrix = reinterpret_cast<int*>(smem_raw + 67040);
    const int size_matrix_addr = smem + 67040;
    int* expert_totals = reinterpret_cast<int*>(smem_raw + 68576);
    const int expert_totals_addr = smem + 68576;
    int* expert_bases = reinterpret_cast<int*>(smem_raw + 68768);
    const int expert_bases_addr = smem + 68768;
    int* source_bases = reinterpret_cast<int*>(smem_raw + 68960);
    const int source_bases_addr = smem + 68960;
    int* push_bases = reinterpret_cast<int*>(smem_raw + 69152);
    const int push_bases_addr = smem + 69152;

    // Mbarrier init (1 barriers at smem_raw[0..8))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // matrix_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int world = mega_pg_world;
    const int rank = mega_pg_rank;
    const int local_experts = 48 / world;
    int route_experts[12];
    float route_scores[12];
    int route_ranks[12];
    if (bid == 2) {
        if (tid == 0) {
            asm volatile("fence.acq_rel.sys;" ::: "memory");
            int x_epoch = metadata_ready[0] + 1;
            #pragma unroll 1
            for (int peer = 0; peer < world; peer++) {
                asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready_peers.ptrs[peer])) + (rank))), "r"(static_cast<unsigned int>((unsigned int)x_epoch)) : "memory");
            }
        }
    }
    if (bid < 1) {
        int grid_thread = bid * 512 + tid;
        #pragma unroll 1
        for (int expert = tid; expert < 49; expert += 512) {
            histogram[expert] = 0;
        }
        int first_0 = grid_thread * 4;
        if (first_0 < 6144) {
            {
                const int4* _ivptr_0 = reinterpret_cast<const int4*>(topk_indices + first_0 + 0);
                int4 _ivld_0;
                _ivld_0 = *_ivptr_0;
                route_experts[0 + 0] = _ivld_0.x;
                route_experts[0 + 1] = _ivld_0.y;
                route_experts[0 + 2] = _ivld_0.z;
                route_experts[0 + 3] = _ivld_0.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(topk_scores + first_0 + 0);
                route_scores[0 + 0] = _v4.x;
                route_scores[0 + 1] = _v4.y;
                route_scores[0 + 2] = _v4.z;
                route_scores[0 + 3] = _v4.w;
            }
        }
        int first_1 = 2048 + grid_thread * 4;
        if (first_1 < 6144) {
            {
                const int4* _ivptr_2 = reinterpret_cast<const int4*>(topk_indices + first_1 + 0);
                int4 _ivld_2;
                _ivld_2 = *_ivptr_2;
                route_experts[4 + 0] = _ivld_2.x;
                route_experts[4 + 1] = _ivld_2.y;
                route_experts[4 + 2] = _ivld_2.z;
                route_experts[4 + 3] = _ivld_2.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(topk_scores + first_1 + 0);
                route_scores[4 + 0] = _v4.x;
                route_scores[4 + 1] = _v4.y;
                route_scores[4 + 2] = _v4.z;
                route_scores[4 + 3] = _v4.w;
            }
        }
        int first_2 = 4096 + grid_thread * 4;
        if (first_2 < 6144) {
            {
                const int4* _ivptr_4 = reinterpret_cast<const int4*>(topk_indices + first_2 + 0);
                int4 _ivld_4;
                _ivld_4 = *_ivptr_4;
                route_experts[8 + 0] = _ivld_4.x;
                route_experts[8 + 1] = _ivld_4.y;
                route_experts[8 + 2] = _ivld_4.z;
                route_experts[8 + 3] = _ivld_4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(topk_scores + first_2 + 0);
                route_scores[8 + 0] = _v4.x;
                route_scores[8 + 1] = _v4.y;
                route_scores[8 + 2] = _v4.z;
                route_scores[8 + 3] = _v4.w;
            }
        }
        int flat_0 = grid_thread * 4;
        if ((unsigned int)route_experts[0] >= 48 || flat_0 / 6 >= 1024) {
            route_experts[0] = 48;
        }
        int flat_1 = grid_thread * 4 + 1;
        if ((unsigned int)route_experts[1] >= 48 || flat_1 / 6 >= 1024) {
            route_experts[1] = 48;
        }
        int flat_2 = grid_thread * 4 + 2;
        if ((unsigned int)route_experts[2] >= 48 || flat_2 / 6 >= 1024) {
            route_experts[2] = 48;
        }
        int flat_3 = grid_thread * 4 + 3;
        if ((unsigned int)route_experts[3] >= 48 || flat_3 / 6 >= 1024) {
            route_experts[3] = 48;
        }
        int flat_4 = grid_thread * 4 + 2048;
        if ((unsigned int)route_experts[4] >= 48 || flat_4 / 6 >= 1024) {
            route_experts[4] = 48;
        }
        int flat_5 = grid_thread * 4 + 2049;
        if ((unsigned int)route_experts[5] >= 48 || flat_5 / 6 >= 1024) {
            route_experts[5] = 48;
        }
        int flat_6 = grid_thread * 4 + 2050;
        if ((unsigned int)route_experts[6] >= 48 || flat_6 / 6 >= 1024) {
            route_experts[6] = 48;
        }
        int flat_7 = grid_thread * 4 + 2051;
        if ((unsigned int)route_experts[7] >= 48 || flat_7 / 6 >= 1024) {
            route_experts[7] = 48;
        }
        int flat_8 = grid_thread * 4 + 4096;
        if ((unsigned int)route_experts[8] >= 48 || flat_8 / 6 >= 1024) {
            route_experts[8] = 48;
        }
        int flat_9 = grid_thread * 4 + 4097;
        if ((unsigned int)route_experts[9] >= 48 || flat_9 / 6 >= 1024) {
            route_experts[9] = 48;
        }
        int flat_10 = grid_thread * 4 + 4098;
        if ((unsigned int)route_experts[10] >= 48 || flat_10 / 6 >= 1024) {
            route_experts[10] = 48;
        }
        int flat_11 = grid_thread * 4 + 4099;
        if ((unsigned int)route_experts[11] >= 48 || flat_11 / 6 >= 1024) {
            route_experts[11] = 48;
        }
        __syncthreads();
        int rank_in_expert_0;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_0) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[0]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[0] = rank_in_expert_0;
        int rank_in_expert_1;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_1) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[1]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[1] = rank_in_expert_1;
        int rank_in_expert_2;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_2) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[2]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[2] = rank_in_expert_2;
        int rank_in_expert_3;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_3) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[3]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[3] = rank_in_expert_3;
        int rank_in_expert_4;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_4) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[4]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[4] = rank_in_expert_4;
        int rank_in_expert_5;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_5) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[5]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[5] = rank_in_expert_5;
        int rank_in_expert_6;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_6) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[6]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[6] = rank_in_expert_6;
        int rank_in_expert_7;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_7) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[7]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[7] = rank_in_expert_7;
        int rank_in_expert_8;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_8) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[8]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[8] = rank_in_expert_8;
        int rank_in_expert_9;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_9) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[9]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[9] = rank_in_expert_9;
        int rank_in_expert_10;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_10) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[10]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[10] = rank_in_expert_10;
        int rank_in_expert_11;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_11) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[11]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[11] = rank_in_expert_11;
        __syncthreads();
        if (warp < world) {
            int size_copy = metadata_ready[0] & 1;
            #pragma unroll 1
            for (int expert_1 = lane; expert_1 < 48; expert_1 += 32) {
                reinterpret_cast<int*>(sizes_by_rank_peers.ptrs[warp])[(size_copy * 8 + rank) * 48 + expert_1] = histogram[expert_1];
            }
        }
        int carry = 0;
        int quad[4];
        int row_0 = tid;
        quad[0] = 0;
        quad[1] = 0;
        quad[2] = 0;
        quad[3] = 0;
        if (row_0 < 12) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&quad[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 3]))
                : "r"(histogram_addr + (unsigned int)(row_0 * 16)));
        }
        int quad_total_0 = quad[0] + quad[1] + quad[2] + quad[3];
        // CtaScanSum: threads=512, warps=16
        int quad_base_0;
        int segment_total_0;
        {
            const uint32_t quad_base_0_own = (uint32_t)(quad_total_0);
            uint32_t quad_base_0_ws = quad_base_0_own;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_0_ws, 1, 32); if (lane >= 1) { quad_base_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_0_ws, 2, 32); if (lane >= 2) { quad_base_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_0_ws, 4, 32); if (lane >= 4) { quad_base_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_0_ws, 8, 32); if (lane >= 8) { quad_base_0_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_0_ws, 16, 32); if (lane >= 16) { quad_base_0_ws += peer; } }
            if (lane == 31) { warp_totals[warp] = (int)quad_base_0_ws; }
            __syncthreads();
            const uint32_t quad_base_0_wt = (lane < 16) ? (uint32_t)warp_totals[lane] : 0;
            uint32_t quad_base_0_wsum = quad_base_0_wt;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_0_wsum, 1, 32); if (lane >= 1) { quad_base_0_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_0_wsum, 2, 32); if (lane >= 2) { quad_base_0_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_0_wsum, 4, 32); if (lane >= 4) { quad_base_0_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_0_wsum, 8, 32); if (lane >= 8) { quad_base_0_wsum += peer; } }
            segment_total_0 = (int)__shfl_sync(0xFFFFFFFF, quad_base_0_wsum, 15);
            const uint32_t quad_base_0_before = __shfl_sync(0xFFFFFFFF, quad_base_0_wsum - quad_base_0_wt, warp);
            quad_base_0 = (int)(quad_base_0_ws - quad_base_0_own + quad_base_0_before);
        }
        if (row_0 < 12) {
            int first_3 = carry + quad_base_0;
            run_start[row_0 * 4] = first_3;
            run_start[row_0 * 4 + 1] = first_3 + quad[0];
            run_start[row_0 * 4 + 2] = first_3 + quad[0] + quad[1];
            run_start[row_0 * 4 + 3] = first_3 + quad[0] + quad[1] + quad[2];
        }
        carry = carry + segment_total_0;
        __syncthreads();
        if (tid == 0) {
            run_start[48] = carry;
        }
        #pragma unroll 1
        for (int expert_2 = tid; expert_2 < 48; expert_2 += 512) {
            source_expert_base[expert_2] = run_start[expert_2];
        }
        __syncthreads();
        int grid_thread_0 = bid * 512 + tid;
        int flat_12 = grid_thread_0 * 4;
        int slot_0 = run_start[route_experts[0]] + route_ranks[0];
        sorted_routes[slot_0] = (unsigned long long)__as_u32(route_scores[0]) << 32ull | (unsigned long long)(unsigned int)flat_12;
        int flat_13 = grid_thread_0 * 4 + 1;
        int slot_1 = run_start[route_experts[1]] + route_ranks[1];
        sorted_routes[slot_1] = (unsigned long long)__as_u32(route_scores[1]) << 32ull | (unsigned long long)(unsigned int)flat_13;
        int flat_14 = grid_thread_0 * 4 + 2;
        int slot_2 = run_start[route_experts[2]] + route_ranks[2];
        sorted_routes[slot_2] = (unsigned long long)__as_u32(route_scores[2]) << 32ull | (unsigned long long)(unsigned int)flat_14;
        int flat_15 = grid_thread_0 * 4 + 3;
        int slot_3 = run_start[route_experts[3]] + route_ranks[3];
        sorted_routes[slot_3] = (unsigned long long)__as_u32(route_scores[3]) << 32ull | (unsigned long long)(unsigned int)flat_15;
        int flat_16 = grid_thread_0 * 4 + 2048;
        int slot_4 = run_start[route_experts[4]] + route_ranks[4];
        sorted_routes[slot_4] = (unsigned long long)__as_u32(route_scores[4]) << 32ull | (unsigned long long)(unsigned int)flat_16;
        int flat_17 = grid_thread_0 * 4 + 2049;
        int slot_5 = run_start[route_experts[5]] + route_ranks[5];
        sorted_routes[slot_5] = (unsigned long long)__as_u32(route_scores[5]) << 32ull | (unsigned long long)(unsigned int)flat_17;
        int flat_18 = grid_thread_0 * 4 + 2050;
        int slot_6 = run_start[route_experts[6]] + route_ranks[6];
        sorted_routes[slot_6] = (unsigned long long)__as_u32(route_scores[6]) << 32ull | (unsigned long long)(unsigned int)flat_18;
        int flat_19 = grid_thread_0 * 4 + 2051;
        int slot_7 = run_start[route_experts[7]] + route_ranks[7];
        sorted_routes[slot_7] = (unsigned long long)__as_u32(route_scores[7]) << 32ull | (unsigned long long)(unsigned int)flat_19;
        int flat_20 = grid_thread_0 * 4 + 4096;
        int slot_8 = run_start[route_experts[8]] + route_ranks[8];
        sorted_routes[slot_8] = (unsigned long long)__as_u32(route_scores[8]) << 32ull | (unsigned long long)(unsigned int)flat_20;
        int flat_21 = grid_thread_0 * 4 + 4097;
        int slot_9 = run_start[route_experts[9]] + route_ranks[9];
        sorted_routes[slot_9] = (unsigned long long)__as_u32(route_scores[9]) << 32ull | (unsigned long long)(unsigned int)flat_21;
        int flat_22 = grid_thread_0 * 4 + 4098;
        int slot_10 = run_start[route_experts[10]] + route_ranks[10];
        sorted_routes[slot_10] = (unsigned long long)__as_u32(route_scores[10]) << 32ull | (unsigned long long)(unsigned int)flat_22;
        int flat_23 = grid_thread_0 * 4 + 4099;
        int slot_11 = run_start[route_experts[11]] + route_ranks[11];
        sorted_routes[slot_11] = (unsigned long long)__as_u32(route_scores[11]) << 32ull | (unsigned long long)(unsigned int)flat_23;
        __syncthreads();
        #pragma unroll 4
        for (int position = tid; position < 6144; position += 512) {
            if (position < carry) {
                unsigned long long element = sorted_routes[position];
                int flat = (int)(unsigned int)(element & 4294967295ull);
                sorted_metadata[position] = (long long)rank << 24ll | (long long)(flat % 6) << 16ll | (long long)(flat / 6) & 65535ll;
                sorted_scores[position] = __uint_as_float((unsigned int)(element >> 32ull));
            }
        }
        __syncthreads();
        if (tid == 0) {
            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sorted_metadata_ready))), "r"(static_cast<unsigned int>(1)) : "memory");
        }
    } else if (bid == 1) {
        int size_copy_1 = metadata_ready[0] & 1;
        #pragma unroll 1
        for (int cell = tid; cell < world * 48; cell += 512) {
            uint32_t cell_words[1];
            do {
                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(cell_words[0]) : "l"(reinterpret_cast<unsigned int*>(sizes_by_rank + (size_copy_1 * 8 * 48 + cell))) : "memory");
            } while (
                (cell_words[0] == 0x80000000u)
            );
            size_matrix[cell] = (int)(unsigned int)cell_words[0];
            sizes_by_rank[(size_copy_1 ^ 1) * 8 * 48 + cell] = -2147483648;
        }
        __syncthreads();
        #pragma unroll 1
        for (int expert_3 = tid; expert_3 < 48; expert_3 += 512) {
            int total = 0;
            if (0 < world) {
                total = total + size_matrix[expert_3];
            }
            if (1 < world) {
                total = total + size_matrix[48 + expert_3];
            }
            if (2 < world) {
                total = total + size_matrix[96 + expert_3];
            }
            if (3 < world) {
                total = total + size_matrix[144 + expert_3];
            }
            if (4 < world) {
                total = total + size_matrix[192 + expert_3];
            }
            if (5 < world) {
                total = total + size_matrix[240 + expert_3];
            }
            if (6 < world) {
                total = total + size_matrix[288 + expert_3];
            }
            if (7 < world) {
                total = total + size_matrix[336 + expert_3];
            }
            expert_totals[expert_3] = (total + 63) / 64 * 64;
            if (expert_3 / local_experts == rank) {
                expert_counts[expert_3 - rank * local_experts] = total;
            }
        }
        __syncthreads();
        int carry_1 = 0;
        int quad_1[4];
        int row_1 = tid;
        quad_1[0] = 0;
        quad_1[1] = 0;
        quad_1[2] = 0;
        quad_1[3] = 0;
        if (row_1 < 12) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&quad_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad_1[(0) + 3]))
                : "r"(expert_totals_addr + (unsigned int)(row_1 * 16)));
        }
        int quad_total_1 = quad_1[0] + quad_1[1] + quad_1[2] + quad_1[3];
        // CtaScanSum: threads=512, warps=16
        int quad_base_1;
        int segment_total_1;
        {
            const uint32_t quad_base_1_own = (uint32_t)(quad_total_1);
            uint32_t quad_base_1_ws = quad_base_1_own;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_1_ws, 1, 32); if (lane >= 1) { quad_base_1_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_1_ws, 2, 32); if (lane >= 2) { quad_base_1_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_1_ws, 4, 32); if (lane >= 4) { quad_base_1_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_1_ws, 8, 32); if (lane >= 8) { quad_base_1_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_1_ws, 16, 32); if (lane >= 16) { quad_base_1_ws += peer; } }
            if (lane == 31) { warp_totals[warp] = (int)quad_base_1_ws; }
            __syncthreads();
            const uint32_t quad_base_1_wt = (lane < 16) ? (uint32_t)warp_totals[lane] : 0;
            uint32_t quad_base_1_wsum = quad_base_1_wt;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_1_wsum, 1, 32); if (lane >= 1) { quad_base_1_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_1_wsum, 2, 32); if (lane >= 2) { quad_base_1_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_1_wsum, 4, 32); if (lane >= 4) { quad_base_1_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_1_wsum, 8, 32); if (lane >= 8) { quad_base_1_wsum += peer; } }
            segment_total_1 = (int)__shfl_sync(0xFFFFFFFF, quad_base_1_wsum, 15);
            const uint32_t quad_base_1_before = __shfl_sync(0xFFFFFFFF, quad_base_1_wsum - quad_base_1_wt, warp);
            quad_base_1 = (int)(quad_base_1_ws - quad_base_1_own + quad_base_1_before);
        }
        if (row_1 < 12) {
            int first_4 = carry_1 + quad_base_1;
            expert_bases[row_1 * 4] = first_4;
            expert_bases[row_1 * 4 + 1] = first_4 + quad_1[0];
            expert_bases[row_1 * 4 + 2] = first_4 + quad_1[0] + quad_1[1];
            expert_bases[row_1 * 4 + 3] = first_4 + quad_1[0] + quad_1[1] + quad_1[2];
        }
        carry_1 = carry_1 + segment_total_1;
        __syncthreads();
        int owner_begin = rank * local_experts;
        #pragma unroll 1
        for (int local_expert = tid; local_expert < local_experts; local_expert += 512) {
            pool_expert_base[local_expert] = expert_bases[owner_begin + local_expert] - expert_bases[owner_begin];
        }
        #pragma unroll 1
        for (int expert_4 = tid; expert_4 < 48; expert_4 += 512) {
            int owner = expert_4 / local_experts;
            int ring_position = (rank - owner + world) % world;
            int destination = expert_bases[expert_4] - expert_bases[owner * local_experts];
            if (0 < ring_position) {
                destination = destination + size_matrix[owner % world * 48 + expert_4];
            }
            if (1 < ring_position) {
                destination = destination + size_matrix[(owner + 1) % world * 48 + expert_4];
            }
            if (2 < ring_position) {
                destination = destination + size_matrix[(owner + 2) % world * 48 + expert_4];
            }
            if (3 < ring_position) {
                destination = destination + size_matrix[(owner + 3) % world * 48 + expert_4];
            }
            if (4 < ring_position) {
                destination = destination + size_matrix[(owner + 4) % world * 48 + expert_4];
            }
            if (5 < ring_position) {
                destination = destination + size_matrix[(owner + 5) % world * 48 + expert_4];
            }
            if (6 < ring_position) {
                destination = destination + size_matrix[(owner + 6) % world * 48 + expert_4];
            }
            if (7 < ring_position) {
                destination = destination + size_matrix[(owner + 7) % world * 48 + expert_4];
            }
            push_destination_base[expert_4] = destination;
        }
        __syncthreads();
        if (tid == 0) {
            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(push_table_ready))), "r"(static_cast<unsigned int>(1)) : "memory");
        }
    }
    if (tid == 0) {
        int _sorted_seen_first;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_sorted_seen_first) : "l"(sorted_metadata_ready) : "memory");
        int _sorted_seen = _sorted_seen_first;
        while (_sorted_seen < 1) {
            asm volatile("nanosleep.u32 %0;" :: "r"(150));
            int _sorted_seen_next;
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_sorted_seen_next) : "l"(sorted_metadata_ready) : "memory");
            _sorted_seen = _sorted_seen_next;
        }
        int _table_seen_first;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_table_seen_first) : "l"(push_table_ready) : "memory");
        int _table_seen = _table_seen_first;
        while (_table_seen < 1) {
            asm volatile("nanosleep.u32 %0;" :: "r"(150));
            int _table_seen_next;
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_table_seen_next) : "l"(push_table_ready) : "memory");
            _table_seen = _table_seen_next;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int expert_5 = tid; expert_5 < 48; expert_5 += 512) {
        source_bases[expert_5] = source_expert_base[expert_5];
        push_bases[expert_5] = push_destination_base[expert_5];
    }
    __syncthreads();
    int routes_total = source_bases[47] + sizes_by_rank[((metadata_ready[0] & 1) * 8 + rank) * 48 + 47];
    int slice_routes = ((routes_total + 47) / 48 + 31) / 32 * 32;
    int slice_begin = (bid * 16 + warp) * slice_routes;
    int slice_end = ((slice_begin + slice_routes < routes_total) ? slice_begin + slice_routes : routes_total);
    int meta_copy = metadata_ready[0] & 1;
    int route_rounds = 0;
    if (slice_begin < routes_total) {
        route_rounds = (slice_end - slice_begin + 31) / 32;
    }
    #pragma unroll 1
    for (int route_round = 0; route_round < route_rounds; route_round++) {
        int route = slice_begin + route_round * 32 + lane;
        if (route < slice_end) {
            int expert_6 = 0;
            int probe_0 = expert_6 + 32;
            if (probe_0 < 48) {
                if (source_bases[probe_0] <= route) {
                    expert_6 = probe_0;
                }
            }
            int probe_1 = expert_6 + 16;
            if (probe_1 < 48) {
                if (source_bases[probe_1] <= route) {
                    expert_6 = probe_1;
                }
            }
            int probe_2 = expert_6 + 8;
            if (probe_2 < 48) {
                if (source_bases[probe_2] <= route) {
                    expert_6 = probe_2;
                }
            }
            int probe_3 = expert_6 + 4;
            if (probe_3 < 48) {
                if (source_bases[probe_3] <= route) {
                    expert_6 = probe_3;
                }
            }
            int probe_4 = expert_6 + 2;
            if (probe_4 < 48) {
                if (source_bases[probe_4] <= route) {
                    expert_6 = probe_4;
                }
            }
            int probe_5 = expert_6 + 1;
            if (probe_5 < 48) {
                if (source_bases[probe_5] <= route) {
                    expert_6 = probe_5;
                }
            }
            int owner_1 = expert_6 / local_experts;
            int slot = push_bases[expert_6] + (route - source_bases[expert_6]);
            long long record = sorted_metadata[route];
            float score = sorted_scores[route];
            unsigned int score_bits = __as_u32(score);
            if (score_bits == 2147483648u) {
                score_bits = 0;
            }
            reinterpret_cast<long long*>(token_src_metadata_peers.ptrs[owner_1])[(slot >> 4 << 5) + (meta_copy << 4) + (slot & 15)] = (long long)((unsigned long long)score_bits << 32ull | (unsigned long long)record);
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
