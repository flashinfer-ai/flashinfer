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
kernel_cake_mega_moe_nvfp4_d69d38f91844574653bb(int* __restrict__ topk_indices, float* __restrict__ topk_scores, long long* sorted_metadata, float* sorted_scores, int* pool_expert_base, int* expert_counts, int* source_expert_base, int* push_destination_base, int* sorted_metadata_ready, int* push_table_ready, int* router_size_counter, int* router_histogram_done, int* source_base_ready, int pool_capacity, int mega_pg_world, int mega_pg_rank, const __grid_constant__ CakePeerPointerTable<unsigned int> mega_pg_flags, int* sizes_by_rank, const __grid_constant__ CakePeerPointerTable<int> sizes_by_rank_peers, int* x_ready, const __grid_constant__ CakePeerPointerTable<int> x_ready_peers, int* sizes, const __grid_constant__ CakePeerPointerTable<int> sizes_peers, int* sizes_ready, const __grid_constant__ CakePeerPointerTable<int> sizes_ready_peers, int* metadata_ready, const __grid_constant__ CakePeerPointerTable<int> metadata_ready_peers, long long* token_src_metadata, const __grid_constant__ CakePeerPointerTable<long long> token_src_metadata_peers, float* fc1_topk_scores, const __grid_constant__ CakePeerPointerTable<float> fc1_topk_scores_peers)
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
    int* dump_base = reinterpret_cast<int*>(smem_raw + 1504);
    const int dump_base_addr = smem + 1504;
    unsigned long long* sorted_routes = reinterpret_cast<unsigned long long*>(smem_raw + 1696);
    const int sorted_routes_addr = smem + 1696;
    int* size_matrix = reinterpret_cast<int*>(smem_raw + 132768);
    const int size_matrix_addr = smem + 132768;
    int* expert_totals = reinterpret_cast<int*>(smem_raw + 134304);
    const int expert_totals_addr = smem + 134304;
    int* expert_bases = reinterpret_cast<int*>(smem_raw + 134496);
    const int expert_bases_addr = smem + 134496;
    int* source_bases = reinterpret_cast<int*>(smem_raw + 134688);
    const int source_bases_addr = smem + 134688;
    int* push_bases = reinterpret_cast<int*>(smem_raw + 134880);
    const int push_bases_addr = smem + 134880;

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
    int route_experts[32];
    float route_scores[32];
    int route_ranks[32];
    if (bid == 23) {
        if (tid == 0) {
            asm volatile("fence.acq_rel.sys;" ::: "memory");
            int x_epoch = metadata_ready[0] + 1;
            #pragma unroll 1
            for (int peer = 0; peer < world; peer++) {
                asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(x_ready_peers.ptrs[peer])) + (rank))), "r"(static_cast<unsigned int>((unsigned int)x_epoch)) : "memory");
            }
        }
    }
    if (bid < 3) {
        int grid_thread = bid * 512 + tid;
        #pragma unroll 1
        for (int expert = tid; expert < 49; expert += 512) {
            histogram[expert] = 0;
        }
        int first_0 = grid_thread * 4;
        if (first_0 < 49152) {
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
        int first_1 = 6144 + grid_thread * 4;
        if (first_1 < 49152) {
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
        int first_2 = 12288 + grid_thread * 4;
        if (first_2 < 49152) {
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
        int first_3 = 18432 + grid_thread * 4;
        if (first_3 < 49152) {
            {
                const int4* _ivptr_6 = reinterpret_cast<const int4*>(topk_indices + first_3 + 0);
                int4 _ivld_6;
                _ivld_6 = *_ivptr_6;
                route_experts[12 + 0] = _ivld_6.x;
                route_experts[12 + 1] = _ivld_6.y;
                route_experts[12 + 2] = _ivld_6.z;
                route_experts[12 + 3] = _ivld_6.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(topk_scores + first_3 + 0);
                route_scores[12 + 0] = _v4.x;
                route_scores[12 + 1] = _v4.y;
                route_scores[12 + 2] = _v4.z;
                route_scores[12 + 3] = _v4.w;
            }
        }
        int first_4 = 24576 + grid_thread * 4;
        if (first_4 < 49152) {
            {
                const int4* _ivptr_8 = reinterpret_cast<const int4*>(topk_indices + first_4 + 0);
                int4 _ivld_8;
                _ivld_8 = *_ivptr_8;
                route_experts[16 + 0] = _ivld_8.x;
                route_experts[16 + 1] = _ivld_8.y;
                route_experts[16 + 2] = _ivld_8.z;
                route_experts[16 + 3] = _ivld_8.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(topk_scores + first_4 + 0);
                route_scores[16 + 0] = _v4.x;
                route_scores[16 + 1] = _v4.y;
                route_scores[16 + 2] = _v4.z;
                route_scores[16 + 3] = _v4.w;
            }
        }
        int first_5 = 30720 + grid_thread * 4;
        if (first_5 < 49152) {
            {
                const int4* _ivptr_10 = reinterpret_cast<const int4*>(topk_indices + first_5 + 0);
                int4 _ivld_10;
                _ivld_10 = *_ivptr_10;
                route_experts[20 + 0] = _ivld_10.x;
                route_experts[20 + 1] = _ivld_10.y;
                route_experts[20 + 2] = _ivld_10.z;
                route_experts[20 + 3] = _ivld_10.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(topk_scores + first_5 + 0);
                route_scores[20 + 0] = _v4.x;
                route_scores[20 + 1] = _v4.y;
                route_scores[20 + 2] = _v4.z;
                route_scores[20 + 3] = _v4.w;
            }
        }
        int first_6 = 36864 + grid_thread * 4;
        if (first_6 < 49152) {
            {
                const int4* _ivptr_12 = reinterpret_cast<const int4*>(topk_indices + first_6 + 0);
                int4 _ivld_12;
                _ivld_12 = *_ivptr_12;
                route_experts[24 + 0] = _ivld_12.x;
                route_experts[24 + 1] = _ivld_12.y;
                route_experts[24 + 2] = _ivld_12.z;
                route_experts[24 + 3] = _ivld_12.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(topk_scores + first_6 + 0);
                route_scores[24 + 0] = _v4.x;
                route_scores[24 + 1] = _v4.y;
                route_scores[24 + 2] = _v4.z;
                route_scores[24 + 3] = _v4.w;
            }
        }
        int first_7 = 43008 + grid_thread * 4;
        if (first_7 < 49152) {
            {
                const int4* _ivptr_14 = reinterpret_cast<const int4*>(topk_indices + first_7 + 0);
                int4 _ivld_14;
                _ivld_14 = *_ivptr_14;
                route_experts[28 + 0] = _ivld_14.x;
                route_experts[28 + 1] = _ivld_14.y;
                route_experts[28 + 2] = _ivld_14.z;
                route_experts[28 + 3] = _ivld_14.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(topk_scores + first_7 + 0);
                route_scores[28 + 0] = _v4.x;
                route_scores[28 + 1] = _v4.y;
                route_scores[28 + 2] = _v4.z;
                route_scores[28 + 3] = _v4.w;
            }
        }
        int flat_0 = grid_thread * 4;
        if ((unsigned int)route_experts[0] >= 48 || flat_0 / 6 >= 8192) {
            route_experts[0] = 48;
        }
        int flat_1 = grid_thread * 4 + 1;
        if ((unsigned int)route_experts[1] >= 48 || flat_1 / 6 >= 8192) {
            route_experts[1] = 48;
        }
        int flat_2 = grid_thread * 4 + 2;
        if ((unsigned int)route_experts[2] >= 48 || flat_2 / 6 >= 8192) {
            route_experts[2] = 48;
        }
        int flat_3 = grid_thread * 4 + 3;
        if ((unsigned int)route_experts[3] >= 48 || flat_3 / 6 >= 8192) {
            route_experts[3] = 48;
        }
        int flat_4 = grid_thread * 4 + 6144;
        if ((unsigned int)route_experts[4] >= 48 || flat_4 / 6 >= 8192) {
            route_experts[4] = 48;
        }
        int flat_5 = grid_thread * 4 + 6145;
        if ((unsigned int)route_experts[5] >= 48 || flat_5 / 6 >= 8192) {
            route_experts[5] = 48;
        }
        int flat_6 = grid_thread * 4 + 6146;
        if ((unsigned int)route_experts[6] >= 48 || flat_6 / 6 >= 8192) {
            route_experts[6] = 48;
        }
        int flat_7 = grid_thread * 4 + 6147;
        if ((unsigned int)route_experts[7] >= 48 || flat_7 / 6 >= 8192) {
            route_experts[7] = 48;
        }
        int flat_8 = grid_thread * 4 + 12288;
        if ((unsigned int)route_experts[8] >= 48 || flat_8 / 6 >= 8192) {
            route_experts[8] = 48;
        }
        int flat_9 = grid_thread * 4 + 12289;
        if ((unsigned int)route_experts[9] >= 48 || flat_9 / 6 >= 8192) {
            route_experts[9] = 48;
        }
        int flat_10 = grid_thread * 4 + 12290;
        if ((unsigned int)route_experts[10] >= 48 || flat_10 / 6 >= 8192) {
            route_experts[10] = 48;
        }
        int flat_11 = grid_thread * 4 + 12291;
        if ((unsigned int)route_experts[11] >= 48 || flat_11 / 6 >= 8192) {
            route_experts[11] = 48;
        }
        int flat_12 = grid_thread * 4 + 18432;
        if ((unsigned int)route_experts[12] >= 48 || flat_12 / 6 >= 8192) {
            route_experts[12] = 48;
        }
        int flat_13 = grid_thread * 4 + 18433;
        if ((unsigned int)route_experts[13] >= 48 || flat_13 / 6 >= 8192) {
            route_experts[13] = 48;
        }
        int flat_14 = grid_thread * 4 + 18434;
        if ((unsigned int)route_experts[14] >= 48 || flat_14 / 6 >= 8192) {
            route_experts[14] = 48;
        }
        int flat_15 = grid_thread * 4 + 18435;
        if ((unsigned int)route_experts[15] >= 48 || flat_15 / 6 >= 8192) {
            route_experts[15] = 48;
        }
        int flat_16 = grid_thread * 4 + 24576;
        if ((unsigned int)route_experts[16] >= 48 || flat_16 / 6 >= 8192) {
            route_experts[16] = 48;
        }
        int flat_17 = grid_thread * 4 + 24577;
        if ((unsigned int)route_experts[17] >= 48 || flat_17 / 6 >= 8192) {
            route_experts[17] = 48;
        }
        int flat_18 = grid_thread * 4 + 24578;
        if ((unsigned int)route_experts[18] >= 48 || flat_18 / 6 >= 8192) {
            route_experts[18] = 48;
        }
        int flat_19 = grid_thread * 4 + 24579;
        if ((unsigned int)route_experts[19] >= 48 || flat_19 / 6 >= 8192) {
            route_experts[19] = 48;
        }
        int flat_20 = grid_thread * 4 + 30720;
        if ((unsigned int)route_experts[20] >= 48 || flat_20 / 6 >= 8192) {
            route_experts[20] = 48;
        }
        int flat_21 = grid_thread * 4 + 30721;
        if ((unsigned int)route_experts[21] >= 48 || flat_21 / 6 >= 8192) {
            route_experts[21] = 48;
        }
        int flat_22 = grid_thread * 4 + 30722;
        if ((unsigned int)route_experts[22] >= 48 || flat_22 / 6 >= 8192) {
            route_experts[22] = 48;
        }
        int flat_23 = grid_thread * 4 + 30723;
        if ((unsigned int)route_experts[23] >= 48 || flat_23 / 6 >= 8192) {
            route_experts[23] = 48;
        }
        int flat_24 = grid_thread * 4 + 36864;
        if ((unsigned int)route_experts[24] >= 48 || flat_24 / 6 >= 8192) {
            route_experts[24] = 48;
        }
        int flat_25 = grid_thread * 4 + 36865;
        if ((unsigned int)route_experts[25] >= 48 || flat_25 / 6 >= 8192) {
            route_experts[25] = 48;
        }
        int flat_26 = grid_thread * 4 + 36866;
        if ((unsigned int)route_experts[26] >= 48 || flat_26 / 6 >= 8192) {
            route_experts[26] = 48;
        }
        int flat_27 = grid_thread * 4 + 36867;
        if ((unsigned int)route_experts[27] >= 48 || flat_27 / 6 >= 8192) {
            route_experts[27] = 48;
        }
        int flat_28 = grid_thread * 4 + 43008;
        if ((unsigned int)route_experts[28] >= 48 || flat_28 / 6 >= 8192) {
            route_experts[28] = 48;
        }
        int flat_29 = grid_thread * 4 + 43009;
        if ((unsigned int)route_experts[29] >= 48 || flat_29 / 6 >= 8192) {
            route_experts[29] = 48;
        }
        int flat_30 = grid_thread * 4 + 43010;
        if ((unsigned int)route_experts[30] >= 48 || flat_30 / 6 >= 8192) {
            route_experts[30] = 48;
        }
        int flat_31 = grid_thread * 4 + 43011;
        if ((unsigned int)route_experts[31] >= 48 || flat_31 / 6 >= 8192) {
            route_experts[31] = 48;
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
        int rank_in_expert_12;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_12) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[12]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[12] = rank_in_expert_12;
        int rank_in_expert_13;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_13) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[13]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[13] = rank_in_expert_13;
        int rank_in_expert_14;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_14) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[14]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[14] = rank_in_expert_14;
        int rank_in_expert_15;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_15) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[15]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[15] = rank_in_expert_15;
        int rank_in_expert_16;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_16) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[16]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[16] = rank_in_expert_16;
        int rank_in_expert_17;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_17) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[17]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[17] = rank_in_expert_17;
        int rank_in_expert_18;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_18) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[18]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[18] = rank_in_expert_18;
        int rank_in_expert_19;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_19) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[19]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[19] = rank_in_expert_19;
        int rank_in_expert_20;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_20) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[20]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[20] = rank_in_expert_20;
        int rank_in_expert_21;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_21) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[21]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[21] = rank_in_expert_21;
        int rank_in_expert_22;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_22) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[22]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[22] = rank_in_expert_22;
        int rank_in_expert_23;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_23) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[23]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[23] = rank_in_expert_23;
        int rank_in_expert_24;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_24) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[24]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[24] = rank_in_expert_24;
        int rank_in_expert_25;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_25) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[25]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[25] = rank_in_expert_25;
        int rank_in_expert_26;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_26) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[26]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[26] = rank_in_expert_26;
        int rank_in_expert_27;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_27) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[27]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[27] = rank_in_expert_27;
        int rank_in_expert_28;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_28) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[28]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[28] = rank_in_expert_28;
        int rank_in_expert_29;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_29) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[29]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[29] = rank_in_expert_29;
        int rank_in_expert_30;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_30) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[30]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[30] = rank_in_expert_30;
        int rank_in_expert_31;
        asm volatile("atom.relaxed.cta.shared::cta.add.s32 %0, [%1], %2;"
            : "=r"(rank_in_expert_31) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&histogram[route_experts[31]]))), "r"(static_cast<int>(1)) : "memory");
        route_ranks[31] = rank_in_expert_31;
        __syncthreads();
        #pragma unroll 1
        for (int expert_1 = tid; expert_1 < 48; expert_1 += 512) {
            int reserved = atomicAdd(&router_size_counter[expert_1], histogram[expert_1]);
            dump_base[expert_1] = reserved;
        }
        __syncthreads();
        if (tid == 0) {
            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(router_histogram_done))), "r"(static_cast<unsigned int>(1)) : "memory");
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
            int first_8 = carry + quad_base_0;
            run_start[row_0 * 4] = first_8;
            run_start[row_0 * 4 + 1] = first_8 + quad[0];
            run_start[row_0 * 4 + 2] = first_8 + quad[0] + quad[1];
            run_start[row_0 * 4 + 3] = first_8 + quad[0] + quad[1] + quad[2];
        }
        carry = carry + segment_total_0;
        __syncthreads();
        if (tid == 0) {
            run_start[48] = carry;
        }
        __syncthreads();
        int grid_thread_0 = bid * 512 + tid;
        int flat_32 = grid_thread_0 * 4;
        int slot_0 = run_start[route_experts[0]] + route_ranks[0];
        sorted_routes[slot_0] = (unsigned long long)__as_u32(route_scores[0]) << 32ull | (unsigned long long)(unsigned int)flat_32;
        int flat_33 = grid_thread_0 * 4 + 1;
        int slot_1 = run_start[route_experts[1]] + route_ranks[1];
        sorted_routes[slot_1] = (unsigned long long)__as_u32(route_scores[1]) << 32ull | (unsigned long long)(unsigned int)flat_33;
        int flat_34 = grid_thread_0 * 4 + 2;
        int slot_2 = run_start[route_experts[2]] + route_ranks[2];
        sorted_routes[slot_2] = (unsigned long long)__as_u32(route_scores[2]) << 32ull | (unsigned long long)(unsigned int)flat_34;
        int flat_35 = grid_thread_0 * 4 + 3;
        int slot_3 = run_start[route_experts[3]] + route_ranks[3];
        sorted_routes[slot_3] = (unsigned long long)__as_u32(route_scores[3]) << 32ull | (unsigned long long)(unsigned int)flat_35;
        int flat_36 = grid_thread_0 * 4 + 6144;
        int slot_4 = run_start[route_experts[4]] + route_ranks[4];
        sorted_routes[slot_4] = (unsigned long long)__as_u32(route_scores[4]) << 32ull | (unsigned long long)(unsigned int)flat_36;
        int flat_37 = grid_thread_0 * 4 + 6145;
        int slot_5 = run_start[route_experts[5]] + route_ranks[5];
        sorted_routes[slot_5] = (unsigned long long)__as_u32(route_scores[5]) << 32ull | (unsigned long long)(unsigned int)flat_37;
        int flat_38 = grid_thread_0 * 4 + 6146;
        int slot_6 = run_start[route_experts[6]] + route_ranks[6];
        sorted_routes[slot_6] = (unsigned long long)__as_u32(route_scores[6]) << 32ull | (unsigned long long)(unsigned int)flat_38;
        int flat_39 = grid_thread_0 * 4 + 6147;
        int slot_7 = run_start[route_experts[7]] + route_ranks[7];
        sorted_routes[slot_7] = (unsigned long long)__as_u32(route_scores[7]) << 32ull | (unsigned long long)(unsigned int)flat_39;
        int flat_40 = grid_thread_0 * 4 + 12288;
        int slot_8 = run_start[route_experts[8]] + route_ranks[8];
        sorted_routes[slot_8] = (unsigned long long)__as_u32(route_scores[8]) << 32ull | (unsigned long long)(unsigned int)flat_40;
        int flat_41 = grid_thread_0 * 4 + 12289;
        int slot_9 = run_start[route_experts[9]] + route_ranks[9];
        sorted_routes[slot_9] = (unsigned long long)__as_u32(route_scores[9]) << 32ull | (unsigned long long)(unsigned int)flat_41;
        int flat_42 = grid_thread_0 * 4 + 12290;
        int slot_10 = run_start[route_experts[10]] + route_ranks[10];
        sorted_routes[slot_10] = (unsigned long long)__as_u32(route_scores[10]) << 32ull | (unsigned long long)(unsigned int)flat_42;
        int flat_43 = grid_thread_0 * 4 + 12291;
        int slot_11 = run_start[route_experts[11]] + route_ranks[11];
        sorted_routes[slot_11] = (unsigned long long)__as_u32(route_scores[11]) << 32ull | (unsigned long long)(unsigned int)flat_43;
        int flat_44 = grid_thread_0 * 4 + 18432;
        int slot_12 = run_start[route_experts[12]] + route_ranks[12];
        sorted_routes[slot_12] = (unsigned long long)__as_u32(route_scores[12]) << 32ull | (unsigned long long)(unsigned int)flat_44;
        int flat_45 = grid_thread_0 * 4 + 18433;
        int slot_13 = run_start[route_experts[13]] + route_ranks[13];
        sorted_routes[slot_13] = (unsigned long long)__as_u32(route_scores[13]) << 32ull | (unsigned long long)(unsigned int)flat_45;
        int flat_46 = grid_thread_0 * 4 + 18434;
        int slot_14 = run_start[route_experts[14]] + route_ranks[14];
        sorted_routes[slot_14] = (unsigned long long)__as_u32(route_scores[14]) << 32ull | (unsigned long long)(unsigned int)flat_46;
        int flat_47 = grid_thread_0 * 4 + 18435;
        int slot_15 = run_start[route_experts[15]] + route_ranks[15];
        sorted_routes[slot_15] = (unsigned long long)__as_u32(route_scores[15]) << 32ull | (unsigned long long)(unsigned int)flat_47;
        int flat_48 = grid_thread_0 * 4 + 24576;
        int slot_16 = run_start[route_experts[16]] + route_ranks[16];
        sorted_routes[slot_16] = (unsigned long long)__as_u32(route_scores[16]) << 32ull | (unsigned long long)(unsigned int)flat_48;
        int flat_49 = grid_thread_0 * 4 + 24577;
        int slot_17 = run_start[route_experts[17]] + route_ranks[17];
        sorted_routes[slot_17] = (unsigned long long)__as_u32(route_scores[17]) << 32ull | (unsigned long long)(unsigned int)flat_49;
        int flat_50 = grid_thread_0 * 4 + 24578;
        int slot_18 = run_start[route_experts[18]] + route_ranks[18];
        sorted_routes[slot_18] = (unsigned long long)__as_u32(route_scores[18]) << 32ull | (unsigned long long)(unsigned int)flat_50;
        int flat_51 = grid_thread_0 * 4 + 24579;
        int slot_19 = run_start[route_experts[19]] + route_ranks[19];
        sorted_routes[slot_19] = (unsigned long long)__as_u32(route_scores[19]) << 32ull | (unsigned long long)(unsigned int)flat_51;
        int flat_52 = grid_thread_0 * 4 + 30720;
        int slot_20 = run_start[route_experts[20]] + route_ranks[20];
        sorted_routes[slot_20] = (unsigned long long)__as_u32(route_scores[20]) << 32ull | (unsigned long long)(unsigned int)flat_52;
        int flat_53 = grid_thread_0 * 4 + 30721;
        int slot_21 = run_start[route_experts[21]] + route_ranks[21];
        sorted_routes[slot_21] = (unsigned long long)__as_u32(route_scores[21]) << 32ull | (unsigned long long)(unsigned int)flat_53;
        int flat_54 = grid_thread_0 * 4 + 30722;
        int slot_22 = run_start[route_experts[22]] + route_ranks[22];
        sorted_routes[slot_22] = (unsigned long long)__as_u32(route_scores[22]) << 32ull | (unsigned long long)(unsigned int)flat_54;
        int flat_55 = grid_thread_0 * 4 + 30723;
        int slot_23 = run_start[route_experts[23]] + route_ranks[23];
        sorted_routes[slot_23] = (unsigned long long)__as_u32(route_scores[23]) << 32ull | (unsigned long long)(unsigned int)flat_55;
        int flat_56 = grid_thread_0 * 4 + 36864;
        int slot_24 = run_start[route_experts[24]] + route_ranks[24];
        sorted_routes[slot_24] = (unsigned long long)__as_u32(route_scores[24]) << 32ull | (unsigned long long)(unsigned int)flat_56;
        int flat_57 = grid_thread_0 * 4 + 36865;
        int slot_25 = run_start[route_experts[25]] + route_ranks[25];
        sorted_routes[slot_25] = (unsigned long long)__as_u32(route_scores[25]) << 32ull | (unsigned long long)(unsigned int)flat_57;
        int flat_58 = grid_thread_0 * 4 + 36866;
        int slot_26 = run_start[route_experts[26]] + route_ranks[26];
        sorted_routes[slot_26] = (unsigned long long)__as_u32(route_scores[26]) << 32ull | (unsigned long long)(unsigned int)flat_58;
        int flat_59 = grid_thread_0 * 4 + 36867;
        int slot_27 = run_start[route_experts[27]] + route_ranks[27];
        sorted_routes[slot_27] = (unsigned long long)__as_u32(route_scores[27]) << 32ull | (unsigned long long)(unsigned int)flat_59;
        int flat_60 = grid_thread_0 * 4 + 43008;
        int slot_28 = run_start[route_experts[28]] + route_ranks[28];
        sorted_routes[slot_28] = (unsigned long long)__as_u32(route_scores[28]) << 32ull | (unsigned long long)(unsigned int)flat_60;
        int flat_61 = grid_thread_0 * 4 + 43009;
        int slot_29 = run_start[route_experts[29]] + route_ranks[29];
        sorted_routes[slot_29] = (unsigned long long)__as_u32(route_scores[29]) << 32ull | (unsigned long long)(unsigned int)flat_61;
        int flat_62 = grid_thread_0 * 4 + 43010;
        int slot_30 = run_start[route_experts[30]] + route_ranks[30];
        sorted_routes[slot_30] = (unsigned long long)__as_u32(route_scores[30]) << 32ull | (unsigned long long)(unsigned int)flat_62;
        int flat_63 = grid_thread_0 * 4 + 43011;
        int slot_31 = run_start[route_experts[31]] + route_ranks[31];
        sorted_routes[slot_31] = (unsigned long long)__as_u32(route_scores[31]) << 32ull | (unsigned long long)(unsigned int)flat_63;
        __syncthreads();
        if (tid == 0) {
            int _seen_first;
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_seen_first) : "l"(source_base_ready) : "memory");
            int _seen = _seen_first;
            while (_seen < 1) {
                asm volatile("nanosleep.u32 %0;" :: "r"(150));
                int _seen_next;
                asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_seen_next) : "l"(source_base_ready) : "memory");
                _seen = _seen_next;
            }
        }
        __syncthreads();
        #pragma unroll 1
        for (int expert_2 = tid; expert_2 < 48; expert_2 += 512) {
            dump_base[expert_2] = dump_base[expert_2] + source_expert_base[expert_2];
        }
        __syncthreads();
        #pragma unroll 1
        for (int expert_3 = warp; expert_3 < 48; expert_3 += 16) {
            int run_begin = run_start[expert_3];
            int run_length = histogram[expert_3];
            int write_begin = dump_base[expert_3];
            #pragma unroll 1
            for (int route_round = 0; route_round < (run_length + 31) / 32; route_round++) {
                int route = route_round * 32 + lane;
                unsigned long long element = sorted_routes[((route < run_length) ? run_begin + route : 0)];
                if (route < run_length) {
                    int flat = (int)(unsigned int)(element & 4294967295ull);
                    sorted_metadata[write_begin + route] = (long long)rank << 24ll | (long long)(flat % 6) << 16ll | (long long)(flat / 6) & 65535ll;
                    sorted_scores[write_begin + route] = __uint_as_float((unsigned int)(element >> 32ull));
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(sorted_metadata_ready))), "r"(static_cast<unsigned int>(1)) : "memory");
        }
    } else if (bid == 3) {
        if (tid == 0) {
            int _seen_1_first;
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_seen_1_first) : "l"(router_histogram_done) : "memory");
            int _seen_1 = _seen_1_first;
            while (_seen_1 < 3) {
                asm volatile("nanosleep.u32 %0;" :: "r"(150));
                int _seen_1_next;
                asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(_seen_1_next) : "l"(router_histogram_done) : "memory");
                _seen_1 = _seen_1_next;
            }
        }
        __syncthreads();
        #pragma unroll 1
        for (int expert_4 = tid; expert_4 < 48; expert_4 += 512) {
            expert_totals[expert_4] = router_size_counter[expert_4];
        }
        __syncthreads();
        if (warp < world) {
            int size_copy = metadata_ready[0] & 1;
            #pragma unroll 1
            for (int expert_5 = lane; expert_5 < 48; expert_5 += 32) {
                reinterpret_cast<int*>(sizes_by_rank_peers.ptrs[warp])[(size_copy * 8 + rank) * 48 + expert_5] = expert_totals[expert_5];
            }
        }
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
            int first_9 = carry_1 + quad_base_1;
            expert_bases[row_1 * 4] = first_9;
            expert_bases[row_1 * 4 + 1] = first_9 + quad_1[0];
            expert_bases[row_1 * 4 + 2] = first_9 + quad_1[0] + quad_1[1];
            expert_bases[row_1 * 4 + 3] = first_9 + quad_1[0] + quad_1[1] + quad_1[2];
        }
        carry_1 = carry_1 + segment_total_1;
        __syncthreads();
        #pragma unroll 1
        for (int expert_6 = tid; expert_6 < 48; expert_6 += 512) {
            source_expert_base[expert_6] = expert_bases[expert_6];
        }
        __syncthreads();
        if (tid == 0) {
            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(source_base_ready))), "r"(static_cast<unsigned int>(1)) : "memory");
        }
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
        for (int expert_7 = tid; expert_7 < 48; expert_7 += 512) {
            int total = 0;
            if (0 < world) {
                total = total + size_matrix[expert_7];
            }
            if (1 < world) {
                total = total + size_matrix[48 + expert_7];
            }
            if (2 < world) {
                total = total + size_matrix[96 + expert_7];
            }
            if (3 < world) {
                total = total + size_matrix[144 + expert_7];
            }
            if (4 < world) {
                total = total + size_matrix[192 + expert_7];
            }
            if (5 < world) {
                total = total + size_matrix[240 + expert_7];
            }
            if (6 < world) {
                total = total + size_matrix[288 + expert_7];
            }
            if (7 < world) {
                total = total + size_matrix[336 + expert_7];
            }
            expert_totals[expert_7] = (total + 63) / 64 * 64;
            if (expert_7 / local_experts == rank) {
                expert_counts[expert_7 - rank * local_experts] = total;
            }
        }
        __syncthreads();
        int carry_0 = 0;
        int quad_0[4];
        int row_2 = tid;
        quad_0[0] = 0;
        quad_0[1] = 0;
        quad_0[2] = 0;
        quad_0[3] = 0;
        if (row_2 < 12) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&quad_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad_0[(0) + 3]))
                : "r"(expert_totals_addr + (unsigned int)(row_2 * 16)));
        }
        int quad_total_2 = quad_0[0] + quad_0[1] + quad_0[2] + quad_0[3];
        // CtaScanSum: threads=512, warps=16
        int quad_base_2;
        int segment_total_2;
        {
            const uint32_t quad_base_2_own = (uint32_t)(quad_total_2);
            uint32_t quad_base_2_ws = quad_base_2_own;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_2_ws, 1, 32); if (lane >= 1) { quad_base_2_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_2_ws, 2, 32); if (lane >= 2) { quad_base_2_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_2_ws, 4, 32); if (lane >= 4) { quad_base_2_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_2_ws, 8, 32); if (lane >= 8) { quad_base_2_ws += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_2_ws, 16, 32); if (lane >= 16) { quad_base_2_ws += peer; } }
            if (lane == 31) { warp_totals[warp] = (int)quad_base_2_ws; }
            __syncthreads();
            const uint32_t quad_base_2_wt = (lane < 16) ? (uint32_t)warp_totals[lane] : 0;
            uint32_t quad_base_2_wsum = quad_base_2_wt;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_2_wsum, 1, 32); if (lane >= 1) { quad_base_2_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_2_wsum, 2, 32); if (lane >= 2) { quad_base_2_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_2_wsum, 4, 32); if (lane >= 4) { quad_base_2_wsum += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, quad_base_2_wsum, 8, 32); if (lane >= 8) { quad_base_2_wsum += peer; } }
            segment_total_2 = (int)__shfl_sync(0xFFFFFFFF, quad_base_2_wsum, 15);
            const uint32_t quad_base_2_before = __shfl_sync(0xFFFFFFFF, quad_base_2_wsum - quad_base_2_wt, warp);
            quad_base_2 = (int)(quad_base_2_ws - quad_base_2_own + quad_base_2_before);
        }
        if (row_2 < 12) {
            int first_10 = carry_0 + quad_base_2;
            expert_bases[row_2 * 4] = first_10;
            expert_bases[row_2 * 4 + 1] = first_10 + quad_0[0];
            expert_bases[row_2 * 4 + 2] = first_10 + quad_0[0] + quad_0[1];
            expert_bases[row_2 * 4 + 3] = first_10 + quad_0[0] + quad_0[1] + quad_0[2];
        }
        carry_0 = carry_0 + segment_total_2;
        __syncthreads();
        int owner_begin = rank * local_experts;
        #pragma unroll 1
        for (int local_expert = tid; local_expert < local_experts; local_expert += 512) {
            pool_expert_base[local_expert] = expert_bases[owner_begin + local_expert] - expert_bases[owner_begin];
        }
        #pragma unroll 1
        for (int expert_8 = tid; expert_8 < 48; expert_8 += 512) {
            int owner = expert_8 / local_experts;
            int ring_position = (rank - owner + world) % world;
            int destination = expert_bases[expert_8] - expert_bases[owner * local_experts];
            if (0 < ring_position) {
                destination = destination + size_matrix[owner % world * 48 + expert_8];
            }
            if (1 < ring_position) {
                destination = destination + size_matrix[(owner + 1) % world * 48 + expert_8];
            }
            if (2 < ring_position) {
                destination = destination + size_matrix[(owner + 2) % world * 48 + expert_8];
            }
            if (3 < ring_position) {
                destination = destination + size_matrix[(owner + 3) % world * 48 + expert_8];
            }
            if (4 < ring_position) {
                destination = destination + size_matrix[(owner + 4) % world * 48 + expert_8];
            }
            if (5 < ring_position) {
                destination = destination + size_matrix[(owner + 5) % world * 48 + expert_8];
            }
            if (6 < ring_position) {
                destination = destination + size_matrix[(owner + 6) % world * 48 + expert_8];
            }
            if (7 < ring_position) {
                destination = destination + size_matrix[(owner + 7) % world * 48 + expert_8];
            }
            push_destination_base[expert_8] = destination;
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
        while (_sorted_seen < 3) {
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
    for (int expert_9 = tid; expert_9 < 48; expert_9 += 512) {
        source_bases[expert_9] = source_expert_base[expert_9];
        push_bases[expert_9] = push_destination_base[expert_9];
    }
    __syncthreads();
    int routes_total = source_bases[47] + sizes_by_rank[((metadata_ready[0] & 1) * 8 + rank) * 48 + 47];
    int slice_routes = ((routes_total + 383) / 384 + 31) / 32 * 32;
    int slice_begin = (bid * 16 + warp) * slice_routes;
    int slice_end = ((slice_begin + slice_routes < routes_total) ? slice_begin + slice_routes : routes_total);
    int meta_copy = metadata_ready[0] & 1;
    int route_rounds = 0;
    if (slice_begin < routes_total) {
        route_rounds = (slice_end - slice_begin + 31) / 32;
    }
    #pragma unroll 1
    for (int route_round_1 = 0; route_round_1 < route_rounds; route_round_1++) {
        int route_1 = slice_begin + route_round_1 * 32 + lane;
        if (route_1 < slice_end) {
            int expert_10 = 0;
            int probe_0 = expert_10 + 32;
            if (probe_0 < 48) {
                if (source_bases[probe_0] <= route_1) {
                    expert_10 = probe_0;
                }
            }
            int probe_1 = expert_10 + 16;
            if (probe_1 < 48) {
                if (source_bases[probe_1] <= route_1) {
                    expert_10 = probe_1;
                }
            }
            int probe_2 = expert_10 + 8;
            if (probe_2 < 48) {
                if (source_bases[probe_2] <= route_1) {
                    expert_10 = probe_2;
                }
            }
            int probe_3 = expert_10 + 4;
            if (probe_3 < 48) {
                if (source_bases[probe_3] <= route_1) {
                    expert_10 = probe_3;
                }
            }
            int probe_4 = expert_10 + 2;
            if (probe_4 < 48) {
                if (source_bases[probe_4] <= route_1) {
                    expert_10 = probe_4;
                }
            }
            int probe_5 = expert_10 + 1;
            if (probe_5 < 48) {
                if (source_bases[probe_5] <= route_1) {
                    expert_10 = probe_5;
                }
            }
            int owner_1 = expert_10 / local_experts;
            int slot = push_bases[expert_10] + (route_1 - source_bases[expert_10]);
            long long record = sorted_metadata[route_1];
            int token = (int)(record & 65535ll);
            int order = 0;
            int count = 0;
            int last = 0;
            int other_0 = topk_indices[token * 6];
            if ((unsigned int)other_0 < 48) {
                if (other_0 / local_experts == owner_1) {
                    count = count + 1;
                    if (other_0 < expert_10) {
                        order = order + 1;
                    }
                    if (other_0 - owner_1 * local_experts > last) {
                        last = other_0 - owner_1 * local_experts;
                    }
                }
            }
            int other_1 = topk_indices[token * 6 + 1];
            if ((unsigned int)other_1 < 48) {
                if (other_1 / local_experts == owner_1) {
                    count = count + 1;
                    if (other_1 < expert_10) {
                        order = order + 1;
                    }
                    if (other_1 - owner_1 * local_experts > last) {
                        last = other_1 - owner_1 * local_experts;
                    }
                }
            }
            int other_2 = topk_indices[token * 6 + 2];
            if ((unsigned int)other_2 < 48) {
                if (other_2 / local_experts == owner_1) {
                    count = count + 1;
                    if (other_2 < expert_10) {
                        order = order + 1;
                    }
                    if (other_2 - owner_1 * local_experts > last) {
                        last = other_2 - owner_1 * local_experts;
                    }
                }
            }
            int other_3 = topk_indices[token * 6 + 3];
            if ((unsigned int)other_3 < 48) {
                if (other_3 / local_experts == owner_1) {
                    count = count + 1;
                    if (other_3 < expert_10) {
                        order = order + 1;
                    }
                    if (other_3 - owner_1 * local_experts > last) {
                        last = other_3 - owner_1 * local_experts;
                    }
                }
            }
            int other_4 = topk_indices[token * 6 + 4];
            if ((unsigned int)other_4 < 48) {
                if (other_4 / local_experts == owner_1) {
                    count = count + 1;
                    if (other_4 < expert_10) {
                        order = order + 1;
                    }
                    if (other_4 - owner_1 * local_experts > last) {
                        last = other_4 - owner_1 * local_experts;
                    }
                }
            }
            int other_5 = topk_indices[token * 6 + 5];
            if ((unsigned int)other_5 < 48) {
                if (other_5 / local_experts == owner_1) {
                    count = count + 1;
                    if (other_5 < expert_10) {
                        order = order + 1;
                    }
                    if (other_5 - owner_1 * local_experts > last) {
                        last = other_5 - owner_1 * local_experts;
                    }
                }
            }
            record = record | (long long)(order | count - 1 << 8 | last << 16) << 32ll;
            float score = sorted_scores[route_1];
            unsigned int score_bits = __as_u32(score);
            if (score_bits == 2147483648u) {
                score_bits = 0;
            }
            reinterpret_cast<long long*>(token_src_metadata_peers.ptrs[owner_1])[(slot >> 4 << 5) + (meta_copy << 4) + (slot & 15)] = record;
            reinterpret_cast<unsigned int*>(fc1_topk_scores_peers.ptrs[owner_1])[(slot >> 5 << 6) + (meta_copy << 5) + (slot & 31)] = score_bits;
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
