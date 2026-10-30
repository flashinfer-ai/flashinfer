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
// clang-format off
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
#define SMEM_TOKEN_BITS_OFF 1024
#define SMEM_TOKEN_BITS_STAGE_BYTES 2064
#define SMEM_TOKEN_BITS_STRIDE 2064
#define SMEM_PLAN_OFFSETS_OFF 3088
#define SMEM_PLAN_OFFSETS_STAGE_BYTES 2064
#define SMEM_PLAN_OFFSETS_STRIDE 2064
#define SMEM_PLAN_COUNTS_OFF 5152
#define SMEM_PLAN_COUNTS_STAGE_BYTES 2048
#define SMEM_PLAN_COUNTS_STRIDE 2048
#define SMEM_PLAN_BASES_OFF 7200
#define SMEM_PLAN_BASES_STAGE_BYTES 2048
#define SMEM_PLAN_BASES_STRIDE 2048
#define SMEM_SCAN_SUMS_OFF 9248
#define SMEM_SCAN_SUMS_STAGE_BYTES 16
#define SMEM_SCAN_SUMS_STRIDE 16
#define SMEM_ROUTE_WEIGHTS_OFF 9264
#define SMEM_ROUTE_WEIGHTS_STAGE_BYTES 512
#define SMEM_ROUTE_WEIGHTS_STRIDE 512
#define SMEM_ROUTE_IDS_OFF 9776
#define SMEM_ROUTE_IDS_STAGE_BYTES 512
#define SMEM_ROUTE_IDS_STRIDE 512
#define SMEM_STAGE_OFF 10288
#define SMEM_STAGE_STAGE_BYTES 4096
#define SMEM_STAGE_STRIDE 4096
#define SMEM_COUNT_TABLE_OFF 14384
#define SMEM_COUNT_TABLE_STAGE_BYTES 8448
#define SMEM_COUNT_TABLE_STRIDE 8448
#define SMEM_TOTAL 22912
#define THREADS 256
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


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}




__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(16,1,1) void
kernel_alpha_moe_fused_router_cluster_c16_w8_threshold(float* __restrict__ logits, float* __restrict__ topk_weights, int* __restrict__ topk_ids, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, int* __restrict__ expert_counts, int* __restrict__ expert_offsets, int* __restrict__ expert_scatter_offsets, int M, int E, int top_k, int block_m, int has_shared_expert)
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
    #define exchange_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 16;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 16;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    unsigned int* token_bits = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int token_bits_addr = smem + 1024;
    int* plan_offsets = reinterpret_cast<int*>(smem_raw + 3088);
    const int plan_offsets_addr = smem + 3088;
    int* plan_counts = reinterpret_cast<int*>(smem_raw + 5152);
    const int plan_counts_addr = smem + 5152;
    int* plan_bases = reinterpret_cast<int*>(smem_raw + 7200);
    const int plan_bases_addr = smem + 7200;
    int* scan_sums = reinterpret_cast<int*>(smem_raw + 9248);
    const int scan_sums_addr = smem + 9248;
    float* route_weights = reinterpret_cast<float*>(smem_raw + 9264);
    const int route_weights_addr = smem + 9264;
    int* route_ids = reinterpret_cast<int*>(smem_raw + 9776);
    const int route_ids_addr = smem + 9776;
    unsigned int* stage = reinterpret_cast<unsigned int*>(smem_raw + 10288);
    const int stage_addr = smem + 10288;
    unsigned int* count_table = reinterpret_cast<unsigned int*>(smem_raw + 14384);
    const int count_table_addr = smem + 14384;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // exchange: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 0, 8448);
        }
    }

    // === Task calls (declaration order) ===
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    int rank = bid;
    int token = rank * 8 + warp;
    unsigned long long row_base = (unsigned long long)token * (unsigned long long)E;
    float row_values[16];
    int selected_expert = 512;
    unsigned int zero_routes = 0;
    unsigned int lanes_below = (1 << (unsigned int)lane) - 1;
    if (token < M) {
        #pragma unroll
        for (int expert_slot_load = 0; expert_slot_load < 16; expert_slot_load++) {
            int expert_load = lane + expert_slot_load * 32;
            row_values[expert_slot_load] = -CUDART_INF_F;
            if (expert_load < E) {
                row_values[expert_slot_load] = logits[row_base + (unsigned long long)expert_load];
            }
        }
    }
    for (int zero_word = tid; zero_word < 516; zero_word += 256) {
        token_bits[zero_word] = 0;
    }
    __syncthreads();
    if (token < M) {
        unsigned int nan_bits = 0;
        float _max_5;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_5) : "f"(row_values[0]), "f"(row_values[1]));
        float _max_4;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_4) : "f"(_max_5), "f"(row_values[2]));
        float _max_7;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_7) : "f"(row_values[3]), "f"(row_values[4]));
        float _max_6;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_6) : "f"(_max_7), "f"(row_values[5]));
        float _max_3;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_3) : "f"(_max_4), "f"(_max_6));
        float _max_9;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_9) : "f"(row_values[6]), "f"(row_values[7]));
        float _max_8;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_8) : "f"(_max_9), "f"(row_values[8]));
        float _max_2;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_2) : "f"(_max_3), "f"(_max_8));
        float _max_13;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_13) : "f"(row_values[9]), "f"(row_values[10]));
        float _max_12;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_12) : "f"(_max_13), "f"(row_values[11]));
        float _max_15;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_15) : "f"(row_values[12]), "f"(row_values[13]));
        float _max_14;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_14) : "f"(_max_15), "f"(row_values[14]));
        float _max_11;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_11) : "f"(_max_12), "f"(_max_14));
        float _max_10;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_10) : "f"(_max_11), "f"(row_values[15]));
        float _max_1;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_1) : "f"(_max_2), "f"(_max_10));
        float _max_0;
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_0) : "f"(_max_1), "f"(-CUDART_INF_F));
        float _bit_cast_src_16 = _max_0;
        nan_bits = reinterpret_cast<unsigned int*>(&_bit_cast_src_16)[0];
        int _warp_vote_17 = __any_sync(0xFFFFFFFF, (nan_bits & 2147483647) > 2139095040);
        if (_warp_vote_17 != 0) {
            #pragma unroll
            for (int nan_slot = 0; nan_slot < 16; nan_slot++) {
                row_values[nan_slot] = fminf(row_values[nan_slot], CUDART_INF_F);
            }
        }
        float selected_logit = -CUDART_INF_F;
        int route_by_route = 1;
        float lane_max_0 = fmaxf(row_values[0], row_values[1]);
        float lane_max_1 = fmaxf(row_values[2], row_values[3]);
        float lane_max_2 = fmaxf(row_values[4], row_values[5]);
        float lane_max_3 = fmaxf(row_values[6], row_values[7]);
        #pragma unroll
        for (int lane_group = 1; lane_group < 2; lane_group++) {
            lane_max_0 = fmaxf(lane_max_0, fmaxf(row_values[lane_group * 8], row_values[lane_group * 8 + 1]));
            lane_max_1 = fmaxf(lane_max_1, fmaxf(row_values[lane_group * 8 + 2], row_values[lane_group * 8 + 3]));
            lane_max_2 = fmaxf(lane_max_2, fmaxf(row_values[lane_group * 8 + 4], row_values[lane_group * 8 + 5]));
            lane_max_3 = fmaxf(lane_max_3, fmaxf(row_values[lane_group * 8 + 6], row_values[lane_group * 8 + 7]));
        }
        unsigned int lane_bits = 0;
        float _bit_cast_src_18 = fmaxf(fmaxf(lane_max_0, lane_max_1), fmaxf(lane_max_2, lane_max_3));
        lane_bits = reinterpret_cast<unsigned int*>(&_bit_cast_src_18)[0];
        if ((lane_bits & 2139095040) == 0) {
            lane_bits = 0;
        }
        unsigned int lane_key = lane_bits ^ 2147483648u;
        if ((lane_bits & 2147483648u) != 0) {
            lane_key = lane_bits ^ 4294967295u;
        }
        int stage_base = warp * 128;
        stage[stage_base + 96 + lane] = lane_key;
        __syncwarp();
        int lanes_above = 0;
        unsigned int lane_keys[16];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[0])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(0) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[4])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(4) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 4) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[8])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(8) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 8) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[12])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys[(12) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 12) * 4)));
        if (lane_keys[0] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[1] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[2] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[3] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[4] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[5] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[6] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[7] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[8] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[9] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[10] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[11] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[12] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[13] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[14] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys[15] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        unsigned int lane_keys_0[16];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(0) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 16) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(4) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 20) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(8) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 24) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lane_keys_0[(12) + 3]))
            : "r"(stage_addr + (unsigned int)((stage_base + 96 + 28) * 4)));
        if (lane_keys_0[0] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[1] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[2] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[3] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[4] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[5] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[6] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[7] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[8] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[9] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[10] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[11] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[12] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[13] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[14] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        if (lane_keys_0[15] > lane_key) {
            lanes_above = lanes_above + 1;
        }
        unsigned int bound_vote = 4294967295;
        if (lanes_above < top_k) {
            bound_vote = lane_key;
        }
        unsigned int bound_key;
        asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(bound_key) : "r"(bound_vote));
        unsigned int bound_bits = bound_key ^ 2147483648u;
        if ((bound_key & 2147483648u) == 0) {
            bound_bits = bound_key ^ 4294967295u;
        }
        float bound_value = 0.0f;
        bound_value = reinterpret_cast<float*>(&bound_bits)[0];
        unsigned int slot_lanes[16];
        int candidates = 0;
        unsigned int _warp_vote_19 = __ballot_sync(0xFFFFFFFF, row_values[0] >= bound_value);
        slot_lanes[0] = _warp_vote_19;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[0]);
        unsigned int _warp_vote_20 = __ballot_sync(0xFFFFFFFF, row_values[1] >= bound_value);
        slot_lanes[1] = _warp_vote_20;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[1]);
        unsigned int _warp_vote_21 = __ballot_sync(0xFFFFFFFF, row_values[2] >= bound_value);
        slot_lanes[2] = _warp_vote_21;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[2]);
        unsigned int _warp_vote_22 = __ballot_sync(0xFFFFFFFF, row_values[3] >= bound_value);
        slot_lanes[3] = _warp_vote_22;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[3]);
        unsigned int _warp_vote_23 = __ballot_sync(0xFFFFFFFF, row_values[4] >= bound_value);
        slot_lanes[4] = _warp_vote_23;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[4]);
        unsigned int _warp_vote_24 = __ballot_sync(0xFFFFFFFF, row_values[5] >= bound_value);
        slot_lanes[5] = _warp_vote_24;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[5]);
        unsigned int _warp_vote_25 = __ballot_sync(0xFFFFFFFF, row_values[6] >= bound_value);
        slot_lanes[6] = _warp_vote_25;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[6]);
        unsigned int _warp_vote_26 = __ballot_sync(0xFFFFFFFF, row_values[7] >= bound_value);
        slot_lanes[7] = _warp_vote_26;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[7]);
        unsigned int _warp_vote_27 = __ballot_sync(0xFFFFFFFF, row_values[8] >= bound_value);
        slot_lanes[8] = _warp_vote_27;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[8]);
        unsigned int _warp_vote_28 = __ballot_sync(0xFFFFFFFF, row_values[9] >= bound_value);
        slot_lanes[9] = _warp_vote_28;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[9]);
        unsigned int _warp_vote_29 = __ballot_sync(0xFFFFFFFF, row_values[10] >= bound_value);
        slot_lanes[10] = _warp_vote_29;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[10]);
        unsigned int _warp_vote_30 = __ballot_sync(0xFFFFFFFF, row_values[11] >= bound_value);
        slot_lanes[11] = _warp_vote_30;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[11]);
        unsigned int _warp_vote_31 = __ballot_sync(0xFFFFFFFF, row_values[12] >= bound_value);
        slot_lanes[12] = _warp_vote_31;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[12]);
        unsigned int _warp_vote_32 = __ballot_sync(0xFFFFFFFF, row_values[13] >= bound_value);
        slot_lanes[13] = _warp_vote_32;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[13]);
        unsigned int _warp_vote_33 = __ballot_sync(0xFFFFFFFF, row_values[14] >= bound_value);
        slot_lanes[14] = _warp_vote_33;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[14]);
        unsigned int _warp_vote_34 = __ballot_sync(0xFFFFFFFF, row_values[15] >= bound_value);
        slot_lanes[15] = _warp_vote_34;
        candidates = candidates + (int)(unsigned int)__popc(slot_lanes[15]);
        if (candidates <= 32 && bound_key != 2147483648u && bound_key > 8388607) {
            route_by_route = 0;
            int candidate_position = 0;
            int stage_rows[16];
            unsigned int stage_word = 0;
            stage_rows[0] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[0] & lanes_below);
            if ((slot_lanes[0] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_35 = row_values[0];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_35)[0];
                stage[stage_rows[0]] = stage_word;
                stage[stage_rows[0] + 32] = (unsigned int)lane;
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[0]);
            stage_rows[1] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[1] & lanes_below);
            if ((slot_lanes[1] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_36 = row_values[1];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_36)[0];
                stage[stage_rows[1]] = stage_word;
                stage[stage_rows[1] + 32] = (unsigned int)(lane + 32);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[1]);
            stage_rows[2] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[2] & lanes_below);
            if ((slot_lanes[2] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_37 = row_values[2];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_37)[0];
                stage[stage_rows[2]] = stage_word;
                stage[stage_rows[2] + 32] = (unsigned int)(lane + 64);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[2]);
            stage_rows[3] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[3] & lanes_below);
            if ((slot_lanes[3] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_38 = row_values[3];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_38)[0];
                stage[stage_rows[3]] = stage_word;
                stage[stage_rows[3] + 32] = (unsigned int)(lane + 96);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[3]);
            stage_rows[4] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[4] & lanes_below);
            if ((slot_lanes[4] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_39 = row_values[4];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_39)[0];
                stage[stage_rows[4]] = stage_word;
                stage[stage_rows[4] + 32] = (unsigned int)(lane + 128);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[4]);
            stage_rows[5] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[5] & lanes_below);
            if ((slot_lanes[5] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_40 = row_values[5];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_40)[0];
                stage[stage_rows[5]] = stage_word;
                stage[stage_rows[5] + 32] = (unsigned int)(lane + 160);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[5]);
            stage_rows[6] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[6] & lanes_below);
            if ((slot_lanes[6] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_41 = row_values[6];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_41)[0];
                stage[stage_rows[6]] = stage_word;
                stage[stage_rows[6] + 32] = (unsigned int)(lane + 192);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[6]);
            stage_rows[7] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[7] & lanes_below);
            if ((slot_lanes[7] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_42 = row_values[7];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_42)[0];
                stage[stage_rows[7]] = stage_word;
                stage[stage_rows[7] + 32] = (unsigned int)(lane + 224);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[7]);
            stage_rows[8] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[8] & lanes_below);
            if ((slot_lanes[8] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_43 = row_values[8];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_43)[0];
                stage[stage_rows[8]] = stage_word;
                stage[stage_rows[8] + 32] = (unsigned int)(lane + 256);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[8]);
            stage_rows[9] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[9] & lanes_below);
            if ((slot_lanes[9] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_44 = row_values[9];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_44)[0];
                stage[stage_rows[9]] = stage_word;
                stage[stage_rows[9] + 32] = (unsigned int)(lane + 288);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[9]);
            stage_rows[10] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[10] & lanes_below);
            if ((slot_lanes[10] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_45 = row_values[10];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_45)[0];
                stage[stage_rows[10]] = stage_word;
                stage[stage_rows[10] + 32] = (unsigned int)(lane + 320);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[10]);
            stage_rows[11] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[11] & lanes_below);
            if ((slot_lanes[11] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_46 = row_values[11];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_46)[0];
                stage[stage_rows[11]] = stage_word;
                stage[stage_rows[11] + 32] = (unsigned int)(lane + 352);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[11]);
            stage_rows[12] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[12] & lanes_below);
            if ((slot_lanes[12] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_47 = row_values[12];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_47)[0];
                stage[stage_rows[12]] = stage_word;
                stage[stage_rows[12] + 32] = (unsigned int)(lane + 384);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[12]);
            stage_rows[13] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[13] & lanes_below);
            if ((slot_lanes[13] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_48 = row_values[13];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_48)[0];
                stage[stage_rows[13]] = stage_word;
                stage[stage_rows[13] + 32] = (unsigned int)(lane + 416);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[13]);
            stage_rows[14] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[14] & lanes_below);
            if ((slot_lanes[14] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_49 = row_values[14];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_49)[0];
                stage[stage_rows[14]] = stage_word;
                stage[stage_rows[14] + 32] = (unsigned int)(lane + 448);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[14]);
            stage_rows[15] = stage_base + candidate_position + (int)(unsigned int)__popc(slot_lanes[15] & lanes_below);
            if ((slot_lanes[15] >> (unsigned int)lane & 1) != 0) {
                float _bit_cast_src_50 = row_values[15];
                stage_word = reinterpret_cast<unsigned int*>(&_bit_cast_src_50)[0];
                stage[stage_rows[15]] = stage_word;
                stage[stage_rows[15] + 32] = (unsigned int)(lane + 480);
            }
            candidate_position = candidate_position + (int)(unsigned int)__popc(slot_lanes[15]);
            __syncwarp();
            unsigned int candidate_bits = 0;
            unsigned int candidate_id = 0;
            unsigned int candidate_key = 0;
            if (candidates > lane) {
                candidate_bits = stage[stage_base + lane];
                candidate_id = stage[stage_base + 32 + lane];
                unsigned int candidate_flat = candidate_bits;
                if ((candidate_bits & 2139095040) == 0) {
                    candidate_flat = 0;
                }
                candidate_key = candidate_flat ^ 2147483648u;
                if ((candidate_flat & 2147483648u) != 0) {
                    candidate_key = candidate_flat ^ 4294967295u;
                }
            }
            stage[stage_base + 96 + lane] = candidate_key;
            __syncwarp();
            int candidate_rank = 0;
            unsigned int candidate_keys[16];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[0])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(0) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[4])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(4) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96 + 4) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[8])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(8) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96 + 8) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[12])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys[(12) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96 + 12) * 4)));
            if (candidate_keys[0] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[1] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[2] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[3] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[4] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[5] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[6] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[7] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[8] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[9] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[10] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[11] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[12] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[13] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[14] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys[15] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            unsigned int candidate_keys_0[16];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(0) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96 + 16) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(4) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96 + 20) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(8) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96 + 24) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&candidate_keys_0[(12) + 3]))
                : "r"(stage_addr + (unsigned int)((stage_base + 96 + 28) * 4)));
            if (candidate_keys_0[0] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[1] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[2] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[3] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[4] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[5] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[6] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[7] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[8] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[9] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[10] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[11] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[12] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[13] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[14] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            if (candidate_keys_0[15] > candidate_key) {
                candidate_rank = candidate_rank + 1;
            }
            unsigned int tied_lanes = __ballot_sync(0xFFFFFFFF, candidates > lane);
            int rank_bit_set = candidate_rank & 1;
            unsigned int rank_bit_lanes = __ballot_sync(0xFFFFFFFF, rank_bit_set != 0);
            if (rank_bit_set == 0) {
                rank_bit_lanes = rank_bit_lanes ^ 4294967295u;
            }
            tied_lanes = tied_lanes & rank_bit_lanes;
            int rank_bit_set_1 = candidate_rank >> 1 & 1;
            unsigned int rank_bit_lanes_2 = __ballot_sync(0xFFFFFFFF, rank_bit_set_1 != 0);
            if (rank_bit_set_1 == 0) {
                rank_bit_lanes_2 = rank_bit_lanes_2 ^ 4294967295u;
            }
            tied_lanes = tied_lanes & rank_bit_lanes_2;
            int rank_bit_set_3 = candidate_rank >> 2 & 1;
            unsigned int rank_bit_lanes_4 = __ballot_sync(0xFFFFFFFF, rank_bit_set_3 != 0);
            if (rank_bit_set_3 == 0) {
                rank_bit_lanes_4 = rank_bit_lanes_4 ^ 4294967295u;
            }
            tied_lanes = tied_lanes & rank_bit_lanes_4;
            int rank_bit_set_5 = candidate_rank >> 3 & 1;
            unsigned int rank_bit_lanes_6 = __ballot_sync(0xFFFFFFFF, rank_bit_set_5 != 0);
            if (rank_bit_set_5 == 0) {
                rank_bit_lanes_6 = rank_bit_lanes_6 ^ 4294967295u;
            }
            tied_lanes = tied_lanes & rank_bit_lanes_6;
            int rank_bit_set_7 = candidate_rank >> 4 & 1;
            unsigned int rank_bit_lanes_8 = __ballot_sync(0xFFFFFFFF, rank_bit_set_7 != 0);
            if (rank_bit_set_7 == 0) {
                rank_bit_lanes_8 = rank_bit_lanes_8 ^ 4294967295u;
            }
            tied_lanes = tied_lanes & rank_bit_lanes_8;
            candidate_rank = candidate_rank + (int)(unsigned int)__popc(tied_lanes & lanes_below);
            if (candidates > lane && candidate_rank < top_k) {
                stage[stage_base + 64 + candidate_rank] = candidate_bits;
                stage[stage_base + 80 + candidate_rank] = candidate_id;
            }
            __syncwarp();
            if (lane < top_k) {
                unsigned int _bit_cast_src_51 = stage[stage_base + 64 + lane];
                selected_logit = reinterpret_cast<float*>(&_bit_cast_src_51)[0];
                selected_expert = (int)stage[stage_base + 80 + lane];
            }
        }
        #pragma unroll 1
        for (int route = 0; route < top_k * route_by_route; route++) {
            float local_max_0 = -CUDART_INF_F;
            float local_max_1 = -CUDART_INF_F;
            float local_max_2 = -CUDART_INF_F;
            float local_max_3 = -CUDART_INF_F;
            #pragma unroll
            for (int scan_group = 0; scan_group < 4; scan_group++) {
                local_max_0 = fmaxf(local_max_0, row_values[scan_group * 4]);
                local_max_1 = fmaxf(local_max_1, row_values[scan_group * 4 + 1]);
                local_max_2 = fmaxf(local_max_2, row_values[scan_group * 4 + 2]);
                local_max_3 = fmaxf(local_max_3, row_values[scan_group * 4 + 3]);
            }
            float local_max_low = fmaxf(local_max_0, local_max_1);
            float local_max_high = fmaxf(local_max_2, local_max_3);
            float local_value = fmaxf(local_max_low, local_max_high);
            unsigned int local_bits = 0;
            local_bits = reinterpret_cast<unsigned int*>(&local_value)[0];
            unsigned int local_key = local_bits ^ 2147483648u;
            if ((local_bits & 2147483648u) != 0) {
                local_key = local_bits ^ 4294967295u;
            }
            unsigned int best_key;
            asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(best_key) : "r"(local_key));
            unsigned int best_bits = best_key ^ 2147483648u;
            if ((best_key & 2147483648u) == 0) {
                best_bits = best_key ^ 4294967295u;
            }
            float best_value = 0.0f;
            best_value = reinterpret_cast<float*>(&best_bits)[0];
            unsigned int tied_index = 512;
            if (best_value != -CUDART_INF_F) {
                #pragma unroll
                for (int expert_slot_finite_tie = 0; expert_slot_finite_tie < 16; expert_slot_finite_tie++) {
                    if (row_values[15 - expert_slot_finite_tie] == best_value) {
                        int expert_finite_tie = lane + (15 - expert_slot_finite_tie) * 32;
                        tied_index = (unsigned int)expert_finite_tie;
                    }
                }
            } else {
                #pragma unroll
                for (int expert_slot_tie = 0; expert_slot_tie < 16; expert_slot_tie++) {
                    int expert_tie = lane + (15 - expert_slot_tie) * 32;
                    if (expert_tie < E) {
                        if (row_values[15 - expert_slot_tie] == best_value) {
                            tied_index = (unsigned int)expert_tie;
                        }
                    }
                }
            }
            unsigned int _warp_redux_52;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_52) : "r"(tied_index));
            int best_index = (int)_warp_redux_52;
            if (lane == route) {
                selected_logit = best_value;
                selected_expert = best_index;
            }
            #pragma unroll
            for (int expert_slot_remove = 0; expert_slot_remove < 16; expert_slot_remove++) {
                if (lane + expert_slot_remove * 32 == best_index) {
                    row_values[expert_slot_remove] = -CUDART_INF_F;
                }
            }
        }
        float selected_for_max = -CUDART_INF_F;
        if (lane < top_k) {
            selected_for_max = selected_logit;
        }
        float selected_max;
        asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(selected_max) : "f"(selected_for_max));
        float selected_exp = 0.0f;
        if (lane < top_k) {
            float _exp2_53 = approx_exp2((selected_logit - selected_max) * 1.442695f);
            selected_exp = _exp2_53;
        }
        float selected_sum = selected_exp;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            selected_sum += __shfl_xor_sync(0xFFFFFFFF, selected_sum, offset);
        float selected_sum_rcp = approx_rcp(selected_sum);
        if (lane < 16) {
            route_weights[warp * 16 + lane] = selected_exp * selected_sum_rcp;
            route_ids[warp * 16 + lane] = selected_expert;
        }
        unsigned int _warp_vote_54 = __ballot_sync(0xFFFFFFFF, lane < top_k && selected_expert == 0);
        zero_routes = _warp_vote_54;
        if (lane < top_k) {
            if (selected_expert != 0 || (zero_routes & lanes_below) == 0) {
                atomicAdd(&token_bits[selected_expert >> 4 << 4 | (selected_expert >> 2 & 3 ^ selected_expert >> 5 & 3) << 2 | selected_expert & 3], 1 << (unsigned int)warp);
            }
        }
        int token_extra = (int)(unsigned int)__popc(zero_routes) - 1;
        if (token_extra < 0) {
            token_extra = 0;
        }
        if (lane < 4) {
            if ((token_extra >> lane & 1) != 0) {
                atomicAdd(&token_bits[512 + lane], 1 << (unsigned int)warp);
            }
        }
    }
    __syncthreads();
    unsigned int cta_extra_routes = 0;
    cta_extra_routes = cta_extra_routes + (unsigned int)(unsigned int)__popc(token_bits[512]);
    cta_extra_routes = cta_extra_routes + ((unsigned int)(unsigned int)__popc(token_bits[513]) << 1);
    cta_extra_routes = cta_extra_routes + ((unsigned int)(unsigned int)__popc(token_bits[514]) << 2);
    cta_extra_routes = cta_extra_routes + ((unsigned int)(unsigned int)__popc(token_bits[515]) << 3);
    unsigned int quad_bits[16];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(0) + 3]))
        : "r"(token_bits_addr + (unsigned int)(lane * 64 + (lane >> 1 & 3 ^ 0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[4])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(4) + 3]))
        : "r"(token_bits_addr + (unsigned int)(lane * 64 + (lane >> 1 & 3 ^ 1) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[8])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(8) + 3]))
        : "r"(token_bits_addr + (unsigned int)(lane * 64 + (lane >> 1 & 3 ^ 2) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[12])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad_bits[(12) + 3]))
        : "r"(token_bits_addr + (unsigned int)(lane * 64 + (lane >> 1 & 3 ^ 3) * 16)));
    unsigned int packed[4];
    packed[0] = (unsigned int)(unsigned int)__popc(quad_bits[0]) | (unsigned int)(unsigned int)__popc(quad_bits[1]) << 8 | (unsigned int)(unsigned int)__popc(quad_bits[2]) << 16 | (unsigned int)(unsigned int)__popc(quad_bits[3]) << 24;
    packed[1] = (unsigned int)(unsigned int)__popc(quad_bits[4]) | (unsigned int)(unsigned int)__popc(quad_bits[5]) << 8 | (unsigned int)(unsigned int)__popc(quad_bits[6]) << 16 | (unsigned int)(unsigned int)__popc(quad_bits[7]) << 24;
    packed[2] = (unsigned int)(unsigned int)__popc(quad_bits[8]) | (unsigned int)(unsigned int)__popc(quad_bits[9]) << 8 | (unsigned int)(unsigned int)__popc(quad_bits[10]) << 16 | (unsigned int)(unsigned int)__popc(quad_bits[11]) << 24;
    packed[3] = (unsigned int)(unsigned int)__popc(quad_bits[12]) | (unsigned int)(unsigned int)__popc(quad_bits[13]) << 8 | (unsigned int)(unsigned int)__popc(quad_bits[14]) << 16 | (unsigned int)(unsigned int)__popc(quad_bits[15]) << 24;
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    #pragma unroll
    for (int peer_round = 0; peer_round < 2; peer_round++) {
        int peer = warp + peer_round * 8;
        if (peer < 16) {
            uint32_t _cluster_mapa_55;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_55) : "r"(count_table_addr + (unsigned int)((rank * 132 + lane * 4) * 4)), "r"(peer));
            uint32_t _mbar_peer_56;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_56) : "r"(exchange_addr), "r"(peer));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_cluster_mapa_55), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]), "r"(_mbar_peer_56) : "memory");
            if (lane == 0) {
                uint32_t _cluster_mapa_57;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_57) : "r"(count_table_addr + (unsigned int)((rank * 132 + 128) * 4)), "r"(peer));
                uint32_t _mbar_peer_58;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_58) : "r"(exchange_addr), "r"(peer));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_cluster_mapa_57), "r"(cta_extra_routes), "r"(0), "r"(0), "r"(0), "r"(_mbar_peer_58) : "memory");
            }
        }
    }
    int cta_tokens = M - rank * 8;
    if (cta_tokens > 8) {
        cta_tokens = 8;
    }
    unsigned long long cta_origin = (unsigned long long)(rank * 8) * (unsigned long long)top_k;
    if (tid >= 128) {
        for (int helper_pair = tid - 128; helper_pair < cta_tokens * top_k; helper_pair += 128) {
            int helper_token = helper_pair / top_k;
            int helper_cell = helper_token * 16 + helper_pair - helper_token * top_k;
            topk_weights[cta_origin + (unsigned long long)helper_pair] = route_weights[helper_cell];
            topk_ids[cta_origin + (unsigned long long)helper_pair] = route_ids[helper_cell];
        }
    }
    if (tid == 0) {
        mbarrier_arrive(exchange_addr);
    }
    asm volatile("{ .reg .pred P1; WAIT: mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 P1, [%0], %1, %2; @P1 bra.uni DONE; bra.uni WAIT; DONE: }" :: "r"(exchange_addr), "r"(uint32_t(0)), "r"(uint32_t(10000000)) : "memory");
    asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
    unsigned int even_total = 0;
    unsigned int odd_total = 0;
    unsigned int even_below = 0;
    unsigned int odd_below = 0;
    int item_counts[4];
    int item_padded[4];
    int thread_prefix[4];
    int thread_total = 0;
    int scan_inclusive = 0;
    int block_rem = block_m & block_m - 1;
    int _find_msb_59;
    asm volatile("bfind.u32 %0, %1;" : "=r"(_find_msb_59) : "r"((unsigned int)block_m));
    int block_shift = (int)_find_msb_59;
    if (tid < 128) {
        #pragma unroll
        for (int source_rank = 0; source_rank < 16; source_rank++) {
            unsigned int source_word = count_table[source_rank * 132 + tid];
            even_total = even_total + (source_word & 16711935);
            odd_total = odd_total + (source_word >> 8 & 16711935);
            if (rank > source_rank) {
                even_below = even_below + (source_word & 16711935);
                odd_below = odd_below + (source_word >> 8 & 16711935);
            }
            if (tid == 0) {
                unsigned int source_extra = count_table[source_rank * 132 + 128];
                even_total = even_total + source_extra;
                if (rank > source_rank) {
                    even_below = even_below + source_extra;
                }
            }
        }
        item_counts[0] = (int)(even_total & 65535);
        item_counts[1] = (int)(odd_total & 65535);
        item_counts[2] = (int)(even_total >> 16);
        item_counts[3] = (int)(odd_total >> 16);
        item_padded[0] = item_counts[0] + block_m - 1 >> block_shift << block_shift;
        item_padded[1] = item_counts[1] + block_m - 1 >> block_shift << block_shift;
        item_padded[2] = item_counts[2] + block_m - 1 >> block_shift << block_shift;
        item_padded[3] = item_counts[3] + block_m - 1 >> block_shift << block_shift;
        if (block_rem != 0) {
            item_padded[0] = (item_counts[0] + block_m - 1) / block_m * block_m;
            item_padded[1] = (item_counts[1] + block_m - 1) / block_m * block_m;
            item_padded[2] = (item_counts[2] + block_m - 1) / block_m * block_m;
            item_padded[3] = (item_counts[3] + block_m - 1) / block_m * block_m;
        }
        thread_prefix[0] = thread_total;
        thread_total = thread_total + item_padded[0];
        thread_prefix[1] = thread_total;
        thread_total = thread_total + item_padded[1];
        thread_prefix[2] = thread_total;
        thread_total = thread_total + item_padded[2];
        thread_prefix[3] = thread_total;
        thread_total = thread_total + item_padded[3];
        uint32_t _warp_scan_sum_60 = (unsigned int)thread_total;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_60, 1, 32); if (lane >= 1) { _warp_scan_sum_60 += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_60, 2, 32); if (lane >= 2) { _warp_scan_sum_60 += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_60, 4, 32); if (lane >= 4) { _warp_scan_sum_60 += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_60, 8, 32); if (lane >= 8) { _warp_scan_sum_60 += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_60, 16, 32); if (lane >= 16) { _warp_scan_sum_60 += peer; } }
        scan_inclusive = (int)_warp_scan_sum_60;
        if (lane == 31) {
            scan_sums[warp] = scan_inclusive;
        }
    }
    __syncthreads();
    if (tid < 128) {
        int warp_prefix = 0;
        #pragma unroll
        for (int prior_warp = 0; prior_warp < 4; prior_warp++) {
            if (prior_warp < warp) {
                warp_prefix = warp_prefix + scan_sums[prior_warp];
            }
        }
        int expert_prefix = warp_prefix + scan_inclusive - thread_total;
        int offset_words[4];
        int base_words[4];
        offset_words[0] = expert_prefix + thread_prefix[0];
        offset_words[1] = expert_prefix + thread_prefix[1];
        offset_words[2] = expert_prefix + thread_prefix[2];
        offset_words[3] = expert_prefix + thread_prefix[3];
        base_words[0] = offset_words[0] + (int)(even_below & 65535);
        base_words[1] = offset_words[1] + (int)(odd_below & 65535);
        base_words[2] = offset_words[2] + (int)(even_below >> 16);
        base_words[3] = offset_words[3] + (int)(odd_below >> 16);
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plan_offsets_addr + (unsigned int)(tid * 16)), "r"(offset_words[0]), "r"(offset_words[1]), "r"(offset_words[2]), "r"(offset_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plan_counts_addr + (unsigned int)(tid * 16)), "r"(item_counts[0]), "r"(item_counts[1]), "r"(item_counts[2]), "r"(item_counts[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(plan_bases_addr + (unsigned int)(tid * 16)), "r"(base_words[0]), "r"(base_words[1]), "r"(base_words[2]), "r"(base_words[3]) : "memory");
        if (tid == 127) {
            plan_offsets[512] = warp_prefix + scan_inclusive;
        }
    }
    __syncthreads();
    if (token < M) {
        if (lane < top_k) {
            unsigned int lower_bits = token_bits[selected_expert >> 4 << 4 | (selected_expert >> 2 & 3 ^ selected_expert >> 5 & 3) << 2 | selected_expert & 3] & (1 << (unsigned int)warp) - 1;
            int pair_row = plan_bases[selected_expert] + (int)(unsigned int)__popc(lower_bits);
            if (selected_expert == 0) {
                pair_row = pair_row + (int)(unsigned int)__popc(zero_routes & lanes_below);
                pair_row = pair_row + (int)(unsigned int)__popc(token_bits[512] & (1 << (unsigned int)warp) - 1);
                pair_row = pair_row + ((int)(unsigned int)__popc(token_bits[513] & (1 << (unsigned int)warp) - 1) << 1);
                pair_row = pair_row + ((int)(unsigned int)__popc(token_bits[514] & (1 << (unsigned int)warp) - 1) << 2);
                pair_row = pair_row + ((int)(unsigned int)__popc(token_bits[515] & (1 << (unsigned int)warp) - 1) << 3);
            }
            sorted_token_ids[pair_row] = token * top_k + lane;
        }
    }
    if (rank == 0 && tid == 0) {
        num_tokens_post_padded[0] = plan_offsets[512];
    }
    int slice_base = rank * 32;
    #pragma unroll
    for (int tail_round = 0; tail_round < 1; tail_round++) {
        int tail_slot = tid + tail_round * 256;
        int tail_expert = slice_base + tail_slot;
        if (tail_slot <= 32 && tail_expert <= E && (tail_slot < 32 || rank == 15)) {
            int tail_start = plan_offsets[tail_expert];
            expert_offsets[tail_expert] = tail_start;
            if (tail_expert < E) {
                int tail_count = plan_counts[tail_expert];
                int tail_end = plan_offsets[tail_expert + 1];
                expert_counts[tail_expert] = tail_count;
                expert_scatter_offsets[tail_expert] = tail_count;
                if (tail_end > tail_start) {
                    #pragma unroll
                    for (int pad_row = 0; pad_row < 16; pad_row++) {
                        if (pad_row < tail_end - tail_start - tail_count) {
                            sorted_token_ids[tail_start + tail_count + pad_row] = M * top_k;
                        }
                    }
                    int first_block = tail_start >> block_shift;
                    int block_count = tail_end - tail_start >> block_shift;
                    if (block_rem != 0) {
                        first_block = tail_start / block_m;
                        block_count = (tail_end - tail_start) / block_m;
                    }
                    #pragma unroll 1
                    for (int tail_block = 0; tail_block < block_count; tail_block++) {
                        expert_ids[first_block + tail_block] = tail_expert;
                    }
                }
            }
        }
    }
}

} // extern "C"
