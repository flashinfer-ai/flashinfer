// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Generated source. Do not edit.
#include <stdint.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap ABI size must remain 128 bytes");

// Keep the exported descriptor ABI independent of the CUDA header alignment.
struct alignas(128) FlashInferTensorMap {
  CUtensorMap value;
};
static_assert(sizeof(FlashInferTensorMap) == 128, "CUtensorMap ABI size must remain 128 bytes");
static_assert(alignof(FlashInferTensorMap) == 128, "CUtensorMap ABI alignment must remain 128 bytes");
__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

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

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

__device__ __forceinline__ uint32_t cvt_bf16x2_e4m3x2(uint16_t src) {
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(dst) : "h"(src));
#else
    uint32_t _f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(src));
    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
    float _f0;
    float _f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(dst) : "f"(_f1), "f"(_f0));
#endif
    return dst;
}

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 64

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_prepare_dispatch(int* __restrict__ send_counters, int* __restrict__ local_token_counter, int ep_size, unsigned int* __restrict__ flag_val, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid < ep_size) {
        send_counters[tid] = 0;
    }
    if (tid == 0) {
        local_token_counter[0] = 0;
        flag_val[0] = flag_val[0] + 1;
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_TARGET_RANKS_OFF 0
#define SMEM_SMEM_TARGET_RANKS_STAGE_BYTES 88
#define SMEM_SMEM_TARGET_RANKS_STRIDE 88
#define SMEM_SMEM_SEND_INDICES_OFF 88
#define SMEM_SMEM_SEND_INDICES_STAGE_BYTES 88
#define SMEM_SMEM_SEND_INDICES_STRIDE 88
#define SMEM_SMEM_LAST_TOKEN_OFF 176
#define SMEM_SMEM_LAST_TOKEN_STAGE_BYTES 4
#define SMEM_SMEM_LAST_TOKEN_STRIDE 4
#define SMEM_TOTAL 256
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_dispatch(int* __restrict__ token_selected_experts, uint8_t* __restrict__ payload_0, uint8_t* __restrict__ payload_1, uint8_t* __restrict__ payload_2, uint8_t* __restrict__ payload_3, uint8_t* __restrict__ payload_4, uint8_t* __restrict__ payload_5, uint8_t* __restrict__ workspace, int* __restrict__ eplb_local_stats, unsigned long long workspace_stride_bytes, unsigned long long flag_val_offset, unsigned long long local_token_counter_offset, unsigned long long send_counters_offset, unsigned long long recv_counters_offset, unsigned long long completion_flags_offset, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long eplb_gathered_stats_offset, unsigned long long payload_0_offset, unsigned long long payload_1_offset, unsigned long long payload_2_offset, unsigned long long payload_3_offset, unsigned long long payload_4_offset, unsigned long long payload_5_offset, int payload_0_bytes, int payload_1_bytes, int payload_2_bytes, int payload_3_bytes, int payload_4_bytes, int payload_5_bytes, int num_payloads, int max_tokens_per_rank, int local_num_tokens, int ep_rank, int ep_size, int num_experts, int top_k, int eplb_stats_num_experts, bool enable_pdl, bool enable_eplb, bool enable_rank_mask, unsigned long long active_rank_mask, int sanitize_payload_index, int invalid_expert_id)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* smem_target_ranks = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_TARGET_RANKS_OFF);
    const int smem_target_ranks_addr = smem + SMEM_SMEM_TARGET_RANKS_OFF;
    int* smem_send_indices = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_SEND_INDICES_OFF);
    const int smem_send_indices_addr = smem + SMEM_SMEM_SEND_INDICES_OFF;
    int* smem_last_token = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_LAST_TOKEN_OFF);
    const int smem_last_token_addr = smem + SMEM_SMEM_LAST_TOKEN_OFF;

    // === Task calls (dependency order) ===
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int local_token_idx = bid;
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    if (local_num_tokens > 0) {
        if (top_k == 6) {
            if (warp == 0) {
                int route = lane;
                int target_rank = -1;
                if (route < 6) {
                    int ep_base = num_experts / ep_size;
                    int ep_remainder = num_experts - ep_base * ep_size;
                    int split = ep_remainder * (ep_base + 1);
                    int route_base = local_token_idx * 6;
                    int expert_id = token_selected_experts[route_base + route];
                    target_rank = 0;
                    if (ep_remainder == 0) {
                        target_rank = expert_id / ep_base;
                    } else if (expert_id < split) {
                        target_rank = expert_id / (ep_base + 1);
                    } else {
                        target_rank = ep_remainder + (expert_id - split) / ep_base;
                    }
                }
                int _shfl_0 = __shfl_sync(0xFFFFFFFF, target_rank, 0);
                int prior_rank_0 = _shfl_0;
                int _shfl_1 = __shfl_sync(0xFFFFFFFF, target_rank, 1);
                int prior_rank_1 = _shfl_1;
                int _shfl_2 = __shfl_sync(0xFFFFFFFF, target_rank, 2);
                int prior_rank_2 = _shfl_2;
                int _shfl_3 = __shfl_sync(0xFFFFFFFF, target_rank, 3);
                int prior_rank_3 = _shfl_3;
                int _shfl_4 = __shfl_sync(0xFFFFFFFF, target_rank, 4);
                int prior_rank_4 = _shfl_4;
                int first_for_rank = ((route < 6) ? 1 : 0);
                if (route > 0 && target_rank == prior_rank_0) {
                    first_for_rank = 0;
                }
                if (route > 1 && target_rank == prior_rank_1) {
                    first_for_rank = 0;
                }
                if (route > 2 && target_rank == prior_rank_2) {
                    first_for_rank = 0;
                }
                if (route > 3 && target_rank == prior_rank_3) {
                    first_for_rank = 0;
                }
                if (route > 4 && target_rank == prior_rank_4) {
                    first_for_rank = 0;
                }
                int stored_rank = -1;
                int send_index = -1;
                if (route < 6) {
                    unsigned long long rank_bit_one = 1;
                    unsigned long long target_bit = rank_bit_one << (unsigned long long)target_rank;
                    int rank_is_active = 1;
                    if (enable_rank_mask) {
                        rank_is_active = (((active_rank_mask & target_bit) != 0) ? 1 : 0);
                    }
                    if (first_for_rank != 0 && rank_is_active != 0) {
                        unsigned long long send_counter_index = (local_workspace_base + send_counters_offset) / 4 + (unsigned long long)target_rank;
                        int _atomic_old_0 = atomicAdd(&workspace_i32[send_counter_index], 1);
                        send_index = _atomic_old_0;
                        stored_rank = target_rank;
                    }
                    unsigned long long topk_workspace_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)local_token_idx * 6;
                    unsigned long long send_index_workspace_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)local_token_idx * 6;
                    smem_target_ranks[route] = stored_rank;
                    smem_send_indices[route] = send_index;
                    workspace_i32[topk_workspace_base + (unsigned long long)route] = stored_rank;
                    workspace_i32[send_index_workspace_base + (unsigned long long)route] = send_index;
                }
            }
        } else if (tid == 0) {
            int ep_base_1 = num_experts / ep_size;
            int ep_remainder_1 = num_experts - ep_base_1 * ep_size;
            int split_1 = ep_remainder_1 * (ep_base_1 + 1);
            unsigned long long seen_ranks = 0;
            int route_base_1 = local_token_idx * top_k;
            unsigned long long topk_workspace_base_1 = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)local_token_idx * (unsigned long long)top_k;
            unsigned long long send_index_workspace_base_1 = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)local_token_idx * (unsigned long long)top_k;
            #pragma unroll 1
            for (int k = 0; k < top_k; k++) {
                int expert_id_1 = token_selected_experts[route_base_1 + k];
                int target_rank_1 = 0;
                if (ep_remainder_1 == 0) {
                    target_rank_1 = expert_id_1 / ep_base_1;
                } else if (expert_id_1 < split_1) {
                    target_rank_1 = expert_id_1 / (ep_base_1 + 1);
                } else {
                    target_rank_1 = ep_remainder_1 + (expert_id_1 - split_1) / ep_base_1;
                }
                unsigned long long rank_bit_one_1 = 1;
                unsigned long long target_bit_1 = rank_bit_one_1 << (unsigned long long)target_rank_1;
                int first_for_rank_1 = (((seen_ranks & target_bit_1) == 0) ? 1 : 0);
                seen_ranks = seen_ranks | target_bit_1;
                int rank_is_active_1 = 1;
                if (enable_rank_mask) {
                    rank_is_active_1 = (((active_rank_mask & target_bit_1) != 0) ? 1 : 0);
                }
                int stored_rank_1 = -1;
                int send_index_1 = -1;
                if (first_for_rank_1 != 0 && rank_is_active_1 != 0) {
                    unsigned long long send_counter_index_1 = (local_workspace_base + send_counters_offset) / 4 + (unsigned long long)target_rank_1;
                    int _atomic_old_1 = atomicAdd(&workspace_i32[send_counter_index_1], 1);
                    send_index_1 = _atomic_old_1;
                    stored_rank_1 = target_rank_1;
                }
                smem_target_ranks[k] = stored_rank_1;
                smem_send_indices[k] = send_index_1;
                workspace_i32[topk_workspace_base_1 + (unsigned long long)k] = stored_rank_1;
                workspace_i32[send_index_workspace_base_1 + (unsigned long long)k] = send_index_1;
            }
        }
        __syncthreads();
        if (num_payloads > 0) {
            unsigned long long source_base = (unsigned long long)local_token_idx * (unsigned long long)payload_0_bytes;
            if (payload_0_bytes % 16 == 0 && payload_0_offset % 16 == 0) {
                int* payload_i32 = reinterpret_cast<int*>(payload_0);
                int* workspace_i32_0 = reinterpret_cast<int*>(workspace);
                #pragma unroll 1
                for (int byte_offset = tid * 16; byte_offset < payload_0_bytes; byte_offset += 4096) {
                    int _vec_load_0[4];
                    {
                        const int4* _ivptr_0 = reinterpret_cast<const int4*>(payload_i32 + (source_base + (unsigned long long)byte_offset) / 4);
                        int4 _ivld_0;
                        _ivld_0 = *_ivptr_0;
                        _vec_load_0[0 + 0] = _ivld_0.x;
                        _vec_load_0[0 + 1] = _ivld_0.y;
                        _vec_load_0[0 + 2] = _ivld_0.z;
                        _vec_load_0[0 + 3] = _ivld_0.w;
                    }
                    #pragma unroll 1
                    for (int k_1 = 0; k_1 < top_k; k_1++) {
                        int target_rank_2 = smem_target_ranks[k_1];
                        int send_index_2 = smem_send_indices[k_1];
                        if (send_index_2 >= 0) {
                            unsigned long long destination = (unsigned long long)target_rank_2 * workspace_stride_bytes + payload_0_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_2) * (unsigned long long)payload_0_bytes + (unsigned long long)byte_offset;
                            {
                                int4 _iv4 = make_int4(_vec_load_0[0 + 0], _vec_load_0[0 + 1], _vec_load_0[0 + 2], _vec_load_0[0 + 3]);
                                *reinterpret_cast<int4*>(workspace_i32_0 + destination / 4) = _iv4;
                            }
                        }
                    }
                }
            } else {
                #pragma unroll 1
                for (int byte_offset_1 = tid; byte_offset_1 < payload_0_bytes; byte_offset_1 += 256) {
                    uint8_t byte_value = payload_0[source_base + (unsigned long long)byte_offset_1];
                    #pragma unroll 1
                    for (int k_2 = 0; k_2 < top_k; k_2++) {
                        int target_rank_3 = smem_target_ranks[k_2];
                        int send_index_3 = smem_send_indices[k_2];
                        if (send_index_3 >= 0) {
                            unsigned long long destination_1 = (unsigned long long)target_rank_3 * workspace_stride_bytes + payload_0_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_3) * (unsigned long long)payload_0_bytes + (unsigned long long)byte_offset_1;
                            workspace[destination_1] = byte_value;
                        }
                    }
                }
            }
        }
        if (num_payloads > 1) {
            unsigned long long source_base_1 = (unsigned long long)local_token_idx * (unsigned long long)payload_1_bytes;
            if (payload_1_bytes % 16 == 0 && payload_1_offset % 16 == 0) {
                int* payload_i32_1 = reinterpret_cast<int*>(payload_1);
                int* workspace_i32_0_1 = reinterpret_cast<int*>(workspace);
                #pragma unroll 1
                for (int byte_offset_2 = tid * 16; byte_offset_2 < payload_1_bytes; byte_offset_2 += 4096) {
                    int _vec_load_1[4];
                    {
                        const int4* _ivptr_1 = reinterpret_cast<const int4*>(payload_i32_1 + (source_base_1 + (unsigned long long)byte_offset_2) / 4);
                        int4 _ivld_1;
                        _ivld_1 = *_ivptr_1;
                        _vec_load_1[0 + 0] = _ivld_1.x;
                        _vec_load_1[0 + 1] = _ivld_1.y;
                        _vec_load_1[0 + 2] = _ivld_1.z;
                        _vec_load_1[0 + 3] = _ivld_1.w;
                    }
                    #pragma unroll 1
                    for (int k_3 = 0; k_3 < top_k; k_3++) {
                        int target_rank_4 = smem_target_ranks[k_3];
                        int send_index_4 = smem_send_indices[k_3];
                        if (send_index_4 >= 0) {
                            unsigned long long destination_2 = (unsigned long long)target_rank_4 * workspace_stride_bytes + payload_1_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_4) * (unsigned long long)payload_1_bytes + (unsigned long long)byte_offset_2;
                            {
                                int4 _iv4 = make_int4(_vec_load_1[0 + 0], _vec_load_1[0 + 1], _vec_load_1[0 + 2], _vec_load_1[0 + 3]);
                                *reinterpret_cast<int4*>(workspace_i32_0_1 + destination_2 / 4) = _iv4;
                            }
                        }
                    }
                }
            } else {
                #pragma unroll 1
                for (int byte_offset_3 = tid; byte_offset_3 < payload_1_bytes; byte_offset_3 += 256) {
                    uint8_t byte_value_1 = payload_1[source_base_1 + (unsigned long long)byte_offset_3];
                    #pragma unroll 1
                    for (int k_4 = 0; k_4 < top_k; k_4++) {
                        int target_rank_5 = smem_target_ranks[k_4];
                        int send_index_5 = smem_send_indices[k_4];
                        if (send_index_5 >= 0) {
                            unsigned long long destination_3 = (unsigned long long)target_rank_5 * workspace_stride_bytes + payload_1_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_5) * (unsigned long long)payload_1_bytes + (unsigned long long)byte_offset_3;
                            workspace[destination_3] = byte_value_1;
                        }
                    }
                }
            }
        }
        if (num_payloads > 2) {
            unsigned long long source_base_2 = (unsigned long long)local_token_idx * (unsigned long long)payload_2_bytes;
            if (payload_2_bytes % 16 == 0 && payload_2_offset % 16 == 0) {
                int* payload_i32_2 = reinterpret_cast<int*>(payload_2);
                int* workspace_i32_0_2 = reinterpret_cast<int*>(workspace);
                #pragma unroll 1
                for (int byte_offset_4 = tid * 16; byte_offset_4 < payload_2_bytes; byte_offset_4 += 4096) {
                    int _vec_load_2[4];
                    {
                        const int4* _ivptr_2 = reinterpret_cast<const int4*>(payload_i32_2 + (source_base_2 + (unsigned long long)byte_offset_4) / 4);
                        int4 _ivld_2;
                        _ivld_2 = *_ivptr_2;
                        _vec_load_2[0 + 0] = _ivld_2.x;
                        _vec_load_2[0 + 1] = _ivld_2.y;
                        _vec_load_2[0 + 2] = _ivld_2.z;
                        _vec_load_2[0 + 3] = _ivld_2.w;
                    }
                    #pragma unroll 1
                    for (int k_5 = 0; k_5 < top_k; k_5++) {
                        int target_rank_6 = smem_target_ranks[k_5];
                        int send_index_6 = smem_send_indices[k_5];
                        if (send_index_6 >= 0) {
                            unsigned long long destination_4 = (unsigned long long)target_rank_6 * workspace_stride_bytes + payload_2_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_6) * (unsigned long long)payload_2_bytes + (unsigned long long)byte_offset_4;
                            {
                                int4 _iv4 = make_int4(_vec_load_2[0 + 0], _vec_load_2[0 + 1], _vec_load_2[0 + 2], _vec_load_2[0 + 3]);
                                *reinterpret_cast<int4*>(workspace_i32_0_2 + destination_4 / 4) = _iv4;
                            }
                        }
                    }
                }
            } else {
                #pragma unroll 1
                for (int byte_offset_5 = tid; byte_offset_5 < payload_2_bytes; byte_offset_5 += 256) {
                    uint8_t byte_value_2 = payload_2[source_base_2 + (unsigned long long)byte_offset_5];
                    #pragma unroll 1
                    for (int k_6 = 0; k_6 < top_k; k_6++) {
                        int target_rank_7 = smem_target_ranks[k_6];
                        int send_index_7 = smem_send_indices[k_6];
                        if (send_index_7 >= 0) {
                            unsigned long long destination_5 = (unsigned long long)target_rank_7 * workspace_stride_bytes + payload_2_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_7) * (unsigned long long)payload_2_bytes + (unsigned long long)byte_offset_5;
                            workspace[destination_5] = byte_value_2;
                        }
                    }
                }
            }
        }
        if (num_payloads > 3) {
            unsigned long long source_base_3 = (unsigned long long)local_token_idx * (unsigned long long)payload_3_bytes;
            if (payload_3_bytes % 16 == 0 && payload_3_offset % 16 == 0) {
                int* payload_i32_3 = reinterpret_cast<int*>(payload_3);
                int* workspace_i32_0_3 = reinterpret_cast<int*>(workspace);
                #pragma unroll 1
                for (int byte_offset_6 = tid * 16; byte_offset_6 < payload_3_bytes; byte_offset_6 += 4096) {
                    int _vec_load_3[4];
                    {
                        const int4* _ivptr_3 = reinterpret_cast<const int4*>(payload_i32_3 + (source_base_3 + (unsigned long long)byte_offset_6) / 4);
                        int4 _ivld_3;
                        _ivld_3 = *_ivptr_3;
                        _vec_load_3[0 + 0] = _ivld_3.x;
                        _vec_load_3[0 + 1] = _ivld_3.y;
                        _vec_load_3[0 + 2] = _ivld_3.z;
                        _vec_load_3[0 + 3] = _ivld_3.w;
                    }
                    #pragma unroll 1
                    for (int k_7 = 0; k_7 < top_k; k_7++) {
                        int target_rank_8 = smem_target_ranks[k_7];
                        int send_index_8 = smem_send_indices[k_7];
                        if (send_index_8 >= 0) {
                            unsigned long long destination_6 = (unsigned long long)target_rank_8 * workspace_stride_bytes + payload_3_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_8) * (unsigned long long)payload_3_bytes + (unsigned long long)byte_offset_6;
                            {
                                int4 _iv4 = make_int4(_vec_load_3[0 + 0], _vec_load_3[0 + 1], _vec_load_3[0 + 2], _vec_load_3[0 + 3]);
                                *reinterpret_cast<int4*>(workspace_i32_0_3 + destination_6 / 4) = _iv4;
                            }
                        }
                    }
                }
            } else {
                #pragma unroll 1
                for (int byte_offset_7 = tid; byte_offset_7 < payload_3_bytes; byte_offset_7 += 256) {
                    uint8_t byte_value_3 = payload_3[source_base_3 + (unsigned long long)byte_offset_7];
                    #pragma unroll 1
                    for (int k_8 = 0; k_8 < top_k; k_8++) {
                        int target_rank_9 = smem_target_ranks[k_8];
                        int send_index_9 = smem_send_indices[k_8];
                        if (send_index_9 >= 0) {
                            unsigned long long destination_7 = (unsigned long long)target_rank_9 * workspace_stride_bytes + payload_3_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_9) * (unsigned long long)payload_3_bytes + (unsigned long long)byte_offset_7;
                            workspace[destination_7] = byte_value_3;
                        }
                    }
                }
            }
        }
        if (num_payloads > 4) {
            unsigned long long source_base_4 = (unsigned long long)local_token_idx * (unsigned long long)payload_4_bytes;
            if (payload_4_bytes % 16 == 0 && payload_4_offset % 16 == 0) {
                int* payload_i32_4 = reinterpret_cast<int*>(payload_4);
                int* workspace_i32_0_4 = reinterpret_cast<int*>(workspace);
                #pragma unroll 1
                for (int byte_offset_8 = tid * 16; byte_offset_8 < payload_4_bytes; byte_offset_8 += 4096) {
                    int _vec_load_4[4];
                    {
                        const int4* _ivptr_4 = reinterpret_cast<const int4*>(payload_i32_4 + (source_base_4 + (unsigned long long)byte_offset_8) / 4);
                        int4 _ivld_4;
                        _ivld_4 = *_ivptr_4;
                        _vec_load_4[0 + 0] = _ivld_4.x;
                        _vec_load_4[0 + 1] = _ivld_4.y;
                        _vec_load_4[0 + 2] = _ivld_4.z;
                        _vec_load_4[0 + 3] = _ivld_4.w;
                    }
                    #pragma unroll 1
                    for (int k_9 = 0; k_9 < top_k; k_9++) {
                        int target_rank_10 = smem_target_ranks[k_9];
                        int send_index_10 = smem_send_indices[k_9];
                        if (send_index_10 >= 0) {
                            unsigned long long destination_8 = (unsigned long long)target_rank_10 * workspace_stride_bytes + payload_4_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_10) * (unsigned long long)payload_4_bytes + (unsigned long long)byte_offset_8;
                            {
                                int4 _iv4 = make_int4(_vec_load_4[0 + 0], _vec_load_4[0 + 1], _vec_load_4[0 + 2], _vec_load_4[0 + 3]);
                                *reinterpret_cast<int4*>(workspace_i32_0_4 + destination_8 / 4) = _iv4;
                            }
                        }
                    }
                }
            } else {
                #pragma unroll 1
                for (int byte_offset_9 = tid; byte_offset_9 < payload_4_bytes; byte_offset_9 += 256) {
                    uint8_t byte_value_4 = payload_4[source_base_4 + (unsigned long long)byte_offset_9];
                    #pragma unroll 1
                    for (int k_10 = 0; k_10 < top_k; k_10++) {
                        int target_rank_11 = smem_target_ranks[k_10];
                        int send_index_11 = smem_send_indices[k_10];
                        if (send_index_11 >= 0) {
                            unsigned long long destination_9 = (unsigned long long)target_rank_11 * workspace_stride_bytes + payload_4_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_11) * (unsigned long long)payload_4_bytes + (unsigned long long)byte_offset_9;
                            workspace[destination_9] = byte_value_4;
                        }
                    }
                }
            }
        }
        if (num_payloads > 5) {
            unsigned long long source_base_5 = (unsigned long long)local_token_idx * (unsigned long long)payload_5_bytes;
            if (payload_5_bytes % 16 == 0 && payload_5_offset % 16 == 0) {
                int* payload_i32_5 = reinterpret_cast<int*>(payload_5);
                int* workspace_i32_0_5 = reinterpret_cast<int*>(workspace);
                #pragma unroll 1
                for (int byte_offset_10 = tid * 16; byte_offset_10 < payload_5_bytes; byte_offset_10 += 4096) {
                    int _vec_load_5[4];
                    {
                        const int4* _ivptr_5 = reinterpret_cast<const int4*>(payload_i32_5 + (source_base_5 + (unsigned long long)byte_offset_10) / 4);
                        int4 _ivld_5;
                        _ivld_5 = *_ivptr_5;
                        _vec_load_5[0 + 0] = _ivld_5.x;
                        _vec_load_5[0 + 1] = _ivld_5.y;
                        _vec_load_5[0 + 2] = _ivld_5.z;
                        _vec_load_5[0 + 3] = _ivld_5.w;
                    }
                    #pragma unroll 1
                    for (int k_11 = 0; k_11 < top_k; k_11++) {
                        int target_rank_12 = smem_target_ranks[k_11];
                        int send_index_12 = smem_send_indices[k_11];
                        if (send_index_12 >= 0) {
                            unsigned long long destination_10 = (unsigned long long)target_rank_12 * workspace_stride_bytes + payload_5_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_12) * (unsigned long long)payload_5_bytes + (unsigned long long)byte_offset_10;
                            {
                                int4 _iv4 = make_int4(_vec_load_5[0 + 0], _vec_load_5[0 + 1], _vec_load_5[0 + 2], _vec_load_5[0 + 3]);
                                *reinterpret_cast<int4*>(workspace_i32_0_5 + destination_10 / 4) = _iv4;
                            }
                        }
                    }
                }
            } else {
                #pragma unroll 1
                for (int byte_offset_11 = tid; byte_offset_11 < payload_5_bytes; byte_offset_11 += 256) {
                    uint8_t byte_value_5 = payload_5[source_base_5 + (unsigned long long)byte_offset_11];
                    #pragma unroll 1
                    for (int k_12 = 0; k_12 < top_k; k_12++) {
                        int target_rank_13 = smem_target_ranks[k_12];
                        int send_index_13 = smem_send_indices[k_12];
                        if (send_index_13 >= 0) {
                            unsigned long long destination_11 = (unsigned long long)target_rank_13 * workspace_stride_bytes + payload_5_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_13) * (unsigned long long)payload_5_bytes + (unsigned long long)byte_offset_11;
                            workspace[destination_11] = byte_value_5;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (warp == 0) {
        int is_last_token = 0;
        if (lane == 0) {
            if (local_num_tokens == 0) {
                is_last_token = 1;
            } else {
                unsigned long long local_counter_index = (local_workspace_base + local_token_counter_offset) / 4;
                int _atomic_old_2 = atomicAdd(&workspace_i32[local_counter_index], 1);
                int completed_before = _atomic_old_2;
                if (completed_before + 1 == local_num_tokens) {
                    is_last_token = 1;
                }
            }
        }
        int _shfl_5 = __shfl_sync(0xFFFFFFFF, is_last_token, 0);
        is_last_token = _shfl_5;
        if (is_last_token != 0) {
            #pragma unroll 1
            for (int target_rank_14 = lane; target_rank_14 < ep_size; target_rank_14 += 32) {
                int rank_is_active_2 = 1;
                if (enable_rank_mask) {
                    unsigned long long completion_bit_one = 1;
                    unsigned long long target_bit_2 = completion_bit_one << (unsigned long long)target_rank_14;
                    rank_is_active_2 = (((active_rank_mask & target_bit_2) != 0) ? 1 : 0);
                }
                if (rank_is_active_2 != 0) {
                    unsigned long long send_counter_index_2 = (local_workspace_base + send_counters_offset) / 4 + (unsigned long long)target_rank_14;
                    unsigned long long recv_counter_index = ((unsigned long long)target_rank_14 * workspace_stride_bytes + recv_counters_offset) / 4 + (unsigned long long)ep_rank;
                    workspace_i32[recv_counter_index] = workspace_i32[send_counter_index_2];
                }
            }
            if (enable_eplb) {
                #pragma unroll 1
                for (int target_rank_eplb = 0; target_rank_eplb < ep_size; target_rank_eplb++) {
                    int rank_is_active_eplb = 1;
                    if (enable_rank_mask) {
                        unsigned long long one_eplb = 1;
                        unsigned long long target_bit_eplb = one_eplb << (unsigned long long)target_rank_eplb;
                        rank_is_active_eplb = (((active_rank_mask & target_bit_eplb) != 0) ? 1 : 0);
                    }
                    if (rank_is_active_eplb != 0) {
                        unsigned long long gathered_base = ((unsigned long long)target_rank_eplb * workspace_stride_bytes + eplb_gathered_stats_offset) / 4 + (unsigned long long)ep_rank * (unsigned long long)eplb_stats_num_experts;
                        #pragma unroll 1
                        for (int expert_stat = lane; expert_stat < eplb_stats_num_experts; expert_stat += 32) {
                            workspace_i32[gathered_base + (unsigned long long)expert_stat] = eplb_local_stats[expert_stat];
                        }
                    }
                }
            }
            unsigned long long expected_value_index = (local_workspace_base + flag_val_offset) / 4;
            unsigned int expected_value = workspace_u32[expected_value_index];
            asm volatile("fence.release.sys;" ::: "memory");
            #pragma unroll 1
            for (int target_rank_flag = lane; target_rank_flag < ep_size; target_rank_flag += 32) {
                int rank_is_active_flag = 1;
                if (enable_rank_mask) {
                    unsigned long long one_flag = 1;
                    unsigned long long target_bit_flag = one_flag << (unsigned long long)target_rank_flag;
                    rank_is_active_flag = (((active_rank_mask & target_bit_flag) != 0) ? 1 : 0);
                }
                if (rank_is_active_flag != 0) {
                    unsigned long long remote_flag_index = ((unsigned long long)target_rank_flag * workspace_stride_bytes + completion_flags_offset) / 4 + (unsigned long long)ep_rank;
                    asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(workspace_u32) + (remote_flag_index))), "r"(static_cast<unsigned int>(expected_value)) : "memory");
                }
            }
            #pragma unroll 1
            for (int peer_rank = lane; peer_rank < ep_size; peer_rank += 32) {
                int peer_is_active = 1;
                if (enable_rank_mask) {
                    unsigned long long one_peer = 1;
                    unsigned long long peer_bit = one_peer << (unsigned long long)peer_rank;
                    peer_is_active = (((active_rank_mask & peer_bit) != 0) ? 1 : 0);
                }
                if (peer_is_active != 0) {
                    unsigned long long local_flag_index = (local_workspace_base + completion_flags_offset) / 4 + (unsigned long long)peer_rank;
                    {
                        unsigned int* _sre_ptr_6 = (reinterpret_cast<unsigned int*>(workspace_u32) + (local_flag_index));
                        const unsigned int _sre_expected_6 = static_cast<unsigned int>(expected_value);
                        const unsigned long long _sre_start_6 = clock64();
                        bool _sre_matched_6 = false;
                        do {
                            unsigned int _sre_value_6;
                            asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(_sre_value_6) : "l"(_sre_ptr_6) : "memory");
                            _sre_matched_6 = (_sre_value_6 == _sre_expected_6);
                        } while (!_sre_matched_6 && ((clock64() - _sre_start_6) <= static_cast<unsigned long long>(600000000000)));
                        if (__builtin_expect(!_sre_matched_6, 0)) {
                            asm volatile("trap;");
                            return;
                        }
                    }
                }
            }
        }
        if (sanitize_payload_index >= 0) {
            if (is_last_token != 0) {
                asm volatile("fence.acquire.sys;" ::: "memory");
            }
            if (lane == 0) {
                smem_last_token[0] = is_last_token;
            }
        }
    }
    if (sanitize_payload_index >= 0) {
        __syncthreads();
        int cta_owns_sanitize = smem_last_token[0];
        if (cta_owns_sanitize != 0) {
            unsigned long long expert_ids_offset = payload_0_offset;
            int expert_ids_bytes = payload_0_bytes;
            if (sanitize_payload_index == 1) {
                expert_ids_offset = payload_1_offset;
                expert_ids_bytes = payload_1_bytes;
            }
            if (sanitize_payload_index == 2) {
                expert_ids_offset = payload_2_offset;
                expert_ids_bytes = payload_2_bytes;
            }
            if (sanitize_payload_index == 3) {
                expert_ids_offset = payload_3_offset;
                expert_ids_bytes = payload_3_bytes;
            }
            if (sanitize_payload_index == 4) {
                expert_ids_offset = payload_4_offset;
                expert_ids_bytes = payload_4_bytes;
            }
            if (sanitize_payload_index == 5) {
                expert_ids_offset = payload_5_offset;
                expert_ids_bytes = payload_5_bytes;
            }
            int row_words = expert_ids_bytes / 4;
            int rank_words = max_tokens_per_rank * row_words;
            unsigned long long expert_ids_base = (local_workspace_base + expert_ids_offset) / 4;
            unsigned long long recv_counters_base = (local_workspace_base + recv_counters_offset) / 4;
            #pragma unroll 1
            for (int source_rank = 0; source_rank < ep_size; source_rank++) {
                int valid_tokens = workspace_i32[recv_counters_base + (unsigned long long)source_rank];
                if (valid_tokens < 0) {
                    valid_tokens = 0;
                }
                int first_word = valid_tokens * row_words;
                unsigned long long rank_base = expert_ids_base + (unsigned long long)source_rank * (unsigned long long)rank_words;
                #pragma unroll 1
                for (int word = first_word + tid; word < rank_words; word += 256) {
                    workspace_i32[rank_base + (unsigned long long)word] = invalid_expert_id;
                }
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_LAST_TOKEN_OFF
#undef SMEM_SMEM_LAST_TOKEN_STAGE_BYTES
#undef SMEM_SMEM_LAST_TOKEN_STRIDE
#undef SMEM_SMEM_SEND_INDICES_OFF
#undef SMEM_SMEM_SEND_INDICES_STAGE_BYTES
#undef SMEM_SMEM_SEND_INDICES_STRIDE
#undef SMEM_SMEM_TARGET_RANKS_OFF
#undef SMEM_SMEM_TARGET_RANKS_STAGE_BYTES
#undef SMEM_SMEM_TARGET_RANKS_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef smem_last_token_addr
#undef smem_send_indices_addr
#undef smem_target_ranks_addr

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_stage_combine(uint8_t* __restrict__ payload, uint8_t* __restrict__ workspace, unsigned long long workspace_stride_bytes, unsigned long long flag_val_offset, unsigned long long destination_offset, unsigned long long payload_bytes, bool advance_generation, int ep_rank, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (advance_generation) {
        if (bid == 0) {
            if (tid == 0) {
                unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
                unsigned long long flag_index = ((unsigned long long)ep_rank * workspace_stride_bytes + flag_val_offset) / 4;
                workspace_u32[flag_index] = workspace_u32[flag_index] + 1;
            }
        }
    }
    unsigned long long linear_thread = (unsigned long long)bid * 256 + (unsigned long long)tid;
    unsigned long long thread_stride = (unsigned long long)num_bids * 256;
    unsigned long long vector_bytes = payload_bytes / 16 * 16;
    int* payload_i32 = reinterpret_cast<int*>(payload);
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    #pragma unroll 1
    for (unsigned long long byte_index = linear_thread * 16; byte_index < vector_bytes; byte_index += thread_stride * 16) {
        int _vec_load_0[4];
        {
            const int4* _ivptr_0 = reinterpret_cast<const int4*>(payload_i32 + byte_index / 4);
            int4 _ivld_0;
            _ivld_0 = *_ivptr_0;
            _vec_load_0[0 + 0] = _ivld_0.x;
            _vec_load_0[0 + 1] = _ivld_0.y;
            _vec_load_0[0 + 2] = _ivld_0.z;
            _vec_load_0[0 + 3] = _ivld_0.w;
        }
        {
            int4 _iv4 = make_int4(_vec_load_0[0 + 0], _vec_load_0[0 + 1], _vec_load_0[0 + 2], _vec_load_0[0 + 3]);
            *reinterpret_cast<int4*>(workspace_i32 + (destination_offset + byte_index) / 4) = _iv4;
        }
    }
    #pragma unroll 1
    for (unsigned long long byte_index_1 = vector_bytes + linear_thread; byte_index_1 < payload_bytes; byte_index_1 += thread_stride) {
        workspace[destination_offset + byte_index_1] = payload[byte_index_1];
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 64

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_publish_combine(uint8_t* __restrict__ workspace, unsigned long long workspace_stride_bytes, unsigned long long flag_val_offset, unsigned long long completion_flags_offset, bool advance_generation, int ep_rank, int ep_size, bool enable_pdl, bool enable_rank_mask, unsigned long long active_rank_mask)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    unsigned long long expected_index = (local_workspace_base + flag_val_offset) / 4;
    if (advance_generation) {
        if (tid == 0) {
            workspace_u32[expected_index] = workspace_u32[expected_index] + 1;
        }
        __syncthreads();
    }
    unsigned int expected_value = workspace_u32[expected_index];
    asm volatile("fence.release.sys;" ::: "memory");
    if (tid < ep_size) {
        int target_rank = tid;
        int target_is_active = 1;
        if (enable_rank_mask) {
            unsigned long long rank_mask_bit = active_rank_mask >> (unsigned long long)target_rank & 1;
            target_is_active = ((rank_mask_bit != 0) ? 1 : 0);
        }
        if (target_is_active != 0) {
            unsigned long long remote_flag_index = ((unsigned long long)target_rank * workspace_stride_bytes + completion_flags_offset) / 4 + (unsigned long long)ep_rank;
            asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(workspace_u32) + (remote_flag_index))), "r"(static_cast<unsigned int>(expected_value)) : "memory");
            unsigned long long local_flag_index = (local_workspace_base + completion_flags_offset) / 4 + (unsigned long long)target_rank;
            {
                unsigned int* _sre_ptr_0 = (reinterpret_cast<unsigned int*>(workspace_u32) + (local_flag_index));
                const unsigned int _sre_expected_0 = static_cast<unsigned int>(expected_value);
                const unsigned long long _sre_start_0 = clock64();
                bool _sre_matched_0 = false;
                do {
                    unsigned int _sre_value_0;
                    asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(_sre_value_0) : "l"(_sre_ptr_0) : "memory");
                    _sre_matched_0 = (_sre_value_0 == _sre_expected_0);
                } while (!_sre_matched_0 && ((clock64() - _sre_start_0) <= static_cast<unsigned long long>(600000000000)));
                if (__builtin_expect(!_sre_matched_0, 0)) {
                    asm volatile("trap;");
                    return;
                }
            }
        }
    }
    __syncthreads();
    asm volatile("fence.acquire.sys;" ::: "memory");
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K 1

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_1(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[TOP_K];
        int send_indices[TOP_K];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            float result = contributions[0];
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K 2

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_2(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[TOP_K];
        int send_indices[TOP_K];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                contributions[0] = contributions[0] + contributions[1];
            }
            float result = contributions[0];
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K 4

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_4(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[TOP_K];
        int send_indices[TOP_K];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                contributions[0] = contributions[0] + contributions[1];
                contributions[2] = contributions[2] + contributions[3];
                contributions[0] = contributions[0] + contributions[2];
            }
            float result = contributions[0];
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K 6

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_6(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[TOP_K];
        int send_indices[TOP_K];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                contributions[0] = contributions[0] + contributions[1];
                contributions[2] = contributions[2] + contributions[3];
                contributions[4] = contributions[4] + contributions[5];
                contributions[0] = contributions[0] + contributions[2];
                contributions[0] = contributions[0] + contributions[4];
            }
            float result = contributions[0];
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K 8

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_8(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[TOP_K];
        int send_indices[TOP_K];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                contributions[0] = contributions[0] + contributions[1];
                contributions[2] = contributions[2] + contributions[3];
                contributions[4] = contributions[4] + contributions[5];
                contributions[6] = contributions[6] + contributions[7];
                contributions[0] = contributions[0] + contributions[2];
                contributions[4] = contributions[4] + contributions[6];
                contributions[0] = contributions[0] + contributions[4];
            }
            float result = contributions[0];
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K 10

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_10(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[TOP_K];
        int send_indices[TOP_K];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                contributions[0] = contributions[0] + contributions[1];
                contributions[2] = contributions[2] + contributions[3];
                contributions[4] = contributions[4] + contributions[5];
                contributions[6] = contributions[6] + contributions[7];
                contributions[8] = contributions[8] + contributions[9];
                contributions[0] = contributions[0] + contributions[2];
                contributions[4] = contributions[4] + contributions[6];
                contributions[0] = contributions[0] + contributions[4];
                contributions[0] = contributions[0] + contributions[8];
            }
            float result = contributions[0];
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K 12

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_12(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[TOP_K];
        int send_indices[TOP_K];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                #pragma unroll
                for (int route_reduce = 1; route_reduce < TOP_K; route_reduce++) {
                    contributions[0] = contributions[0] + contributions[route_reduce];
                }
            }
            float result = contributions[0];
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K 14

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_14(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[TOP_K];
        int send_indices[TOP_K];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                #pragma unroll
                for (int route_reduce = 1; route_reduce < TOP_K; route_reduce++) {
                    contributions[0] = contributions[0] + contributions[route_reduce];
                }
            }
            float result = contributions[0];
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K 16

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_16(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[TOP_K];
        int send_indices[TOP_K];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                contributions[0] = contributions[0] + contributions[1];
                contributions[2] = contributions[2] + contributions[3];
                contributions[4] = contributions[4] + contributions[5];
                contributions[6] = contributions[6] + contributions[7];
                contributions[8] = contributions[8] + contributions[9];
                contributions[10] = contributions[10] + contributions[11];
                contributions[12] = contributions[12] + contributions[13];
                contributions[14] = contributions[14] + contributions[15];
                contributions[0] = contributions[0] + contributions[2];
                contributions[4] = contributions[4] + contributions[6];
                contributions[8] = contributions[8] + contributions[10];
                contributions[12] = contributions[12] + contributions[14];
                contributions[0] = contributions[0] + contributions[4];
                contributions[8] = contributions[8] + contributions[12];
                contributions[0] = contributions[0] + contributions[8];
            }
            float result = contributions[0];
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K 18

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_18(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[TOP_K];
        int send_indices[TOP_K];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                #pragma unroll
                for (int route_reduce = 1; route_reduce < TOP_K; route_reduce++) {
                    contributions[0] = contributions[0] + contributions[route_reduce];
                }
            }
            float result = contributions[0];
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K 22

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_22(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[TOP_K];
        int send_indices[TOP_K];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    float value = 0.0f;
                    if (payload_dtype_code == 0) {
                        __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
                        value = (float)source_bf16[byte_index / 2];
                    }
                    if (payload_dtype_code == 1) {
                        __half* source_f16 = reinterpret_cast<__half*>(workspace);
                        value = (float)source_f16[byte_index / 2];
                    }
                    if (payload_dtype_code == 2) {
                        uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(workspace);
                        float _vec_load_0[1];
                        {
                            uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + byte_index);
                            uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                            uint32_t _f16x2_0;
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                            uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
                        }
                        value = _vec_load_0[0];
                    }
                    if (payload_dtype_code == 3) {
                        float* source_f32 = reinterpret_cast<float*>(workspace);
                        value = source_f32[byte_index / 4];
                    }
                    float contribution = value;
                    if (use_low_precision) {
                        float _fp8_rt_0;
                        uint16_t _e4m3x2_1;
                        uint32_t _f16x2_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                contributions[0] = contributions[0] + contributions[1];
                contributions[2] = contributions[2] + contributions[3];
                contributions[4] = contributions[4] + contributions[5];
                contributions[6] = contributions[6] + contributions[7];
                contributions[8] = contributions[8] + contributions[9];
                contributions[10] = contributions[10] + contributions[11];
                contributions[12] = contributions[12] + contributions[13];
                contributions[14] = contributions[14] + contributions[15];
                contributions[16] = contributions[16] + contributions[17];
                contributions[18] = contributions[18] + contributions[19];
                contributions[20] = contributions[20] + contributions[21];
                contributions[0] = contributions[0] + contributions[2];
                contributions[4] = contributions[4] + contributions[6];
                contributions[8] = contributions[8] + contributions[10];
                contributions[12] = contributions[12] + contributions[14];
                contributions[16] = contributions[16] + contributions[18];
                contributions[0] = contributions[0] + contributions[4];
                contributions[8] = contributions[8] + contributions[12];
                contributions[16] = contributions[16] + contributions[20];
                contributions[0] = contributions[0] + contributions[8];
                contributions[0] = contributions[0] + contributions[16];
            }
            float result = contributions[0];
            unsigned long long output_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
            if (output_dtype_code == 0) {
                __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
                output_bf16[output_item] = result;
            }
            if (output_dtype_code == 1) {
                __half* output_f16 = reinterpret_cast<__half*>(output);
                output_f16[output_item] = result;
            }
            if (output_dtype_code == 2) {
                uint8_t* output_fp8 = reinterpret_cast<uint8_t*>(output);
                {
                    unsigned short _fp8_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(result));
                    *(reinterpret_cast<unsigned char*>(output_fp8 + output_item) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                }
            }
            if (output_dtype_code == 3) {
                float* output_f32 = reinterpret_cast<float*>(output);
                output_f32[output_item] = result;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K

extern "C" {

__global__ __launch_bounds__(32) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_topk6(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long flag_val_offset, unsigned long long completion_flags_offset, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, int ep_size, bool use_low_precision, bool enable_pdl, bool fuse_publication, bool enable_rank_mask, unsigned long long active_rank_mask)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    __nv_bfloat16* workspace_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
    __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    if (fuse_publication) {
        unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
        unsigned long long expected_index = (local_workspace_base + flag_val_offset) / 4ull;
        unsigned int expected_value = workspace_u32[expected_index];
        if (bid == 0 && blockIdx.y == 0) {
            if (tid < 32) {
                asm volatile("fence.release.sys;" ::: "memory");
                #pragma unroll 1
                for (int target_rank = lane; target_rank < ep_size; target_rank += 32) {
                    int target_is_active = 1;
                    if (enable_rank_mask) {
                        unsigned long long target_bit = active_rank_mask >> (unsigned long long)target_rank & 1ull;
                        target_is_active = ((target_bit != 0ull) ? 1 : 0);
                    }
                    if (target_is_active != 0) {
                        unsigned long long remote_flag_index = ((unsigned long long)target_rank * workspace_stride_bytes + completion_flags_offset) / 4ull + (unsigned long long)ep_rank;
                        asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(workspace_u32) + (remote_flag_index))), "r"(static_cast<unsigned int>(expected_value)) : "memory");
                    }
                }
            }
        }
        if (tid < 32) {
            #pragma unroll 1
            for (int peer_rank = lane; peer_rank < ep_size; peer_rank += 32) {
                int peer_is_active = 1;
                if (enable_rank_mask) {
                    unsigned long long peer_bit = active_rank_mask >> (unsigned long long)peer_rank & 1ull;
                    peer_is_active = ((peer_bit != 0ull) ? 1 : 0);
                }
                if (peer_is_active != 0) {
                    unsigned long long local_flag_index = (local_workspace_base + completion_flags_offset) / 4ull + (unsigned long long)peer_rank;
                    {
                        unsigned int* _sre_ptr_0 = (reinterpret_cast<unsigned int*>(workspace_u32) + (local_flag_index));
                        const unsigned int _sre_expected_0 = static_cast<unsigned int>(expected_value);
                        const unsigned long long _sre_start_0 = clock64();
                        bool _sre_matched_0 = false;
                        do {
                            unsigned int _sre_value_0;
                            asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(_sre_value_0) : "l"(_sre_ptr_0) : "memory");
                            _sre_matched_0 = (_sre_value_0 == _sre_expected_0);
                        } while (!_sre_matched_0 && ((clock64() - _sre_start_0) <= static_cast<unsigned long long>(600000000000)));
                        if (__builtin_expect(!_sre_matched_0, 0)) {
                            asm volatile("trap;");
                        }
                    }
                }
            }
            asm volatile("fence.acquire.sys;" ::: "memory");
        }
        __syncthreads();
    }
    int token = bid;
    int column_chunk = blockIdx.y * 256;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * 6ull;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * 6ull;
        unsigned long long lane_route_word_base_plus_one = 0;
        if (lane < 6) {
            int target_rank_1 = workspace_i32[target_base + (unsigned long long)lane];
            int send_index = workspace_i32[send_base + (unsigned long long)lane];
            if (target_rank_1 >= 0 && send_index >= 0) {
                unsigned long long payload_row = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token;
                lane_route_word_base_plus_one = ((unsigned long long)target_rank_1 * workspace_stride_bytes + combine_payload_offset + payload_row * (unsigned long long)payload_element_bytes) / 2ull + 1ull;
            }
        }
        int column = column_chunk + tid * 8;
        unsigned int route_carriers[24];
        route_carriers[0] = 0;
        route_carriers[1] = 0;
        route_carriers[2] = 0;
        route_carriers[3] = 0;
        route_carriers[4] = 0;
        route_carriers[5] = 0;
        route_carriers[6] = 0;
        route_carriers[7] = 0;
        route_carriers[8] = 0;
        route_carriers[9] = 0;
        route_carriers[10] = 0;
        route_carriers[11] = 0;
        route_carriers[12] = 0;
        route_carriers[13] = 0;
        route_carriers[14] = 0;
        route_carriers[15] = 0;
        route_carriers[16] = 0;
        route_carriers[17] = 0;
        route_carriers[18] = 0;
        route_carriers[19] = 0;
        route_carriers[20] = 0;
        route_carriers[21] = 0;
        route_carriers[22] = 0;
        route_carriers[23] = 0;
        #pragma unroll
        for (int route = 0; route < 6; route++) {
            unsigned long long route_word_base_plus_one = __shfl_sync(0xFFFFFFFF, lane_route_word_base_plus_one, route);
            if (column < elements_per_token && route_word_base_plus_one != 0ull) {
                unsigned long long word_index = route_word_base_plus_one - 1ull + (unsigned long long)column;
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(workspace_bf16 + word_index);
                    uint4* _vdst_1 = reinterpret_cast<uint4*>(&route_carriers[route * 4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_1[_blk] = _vptr_1[_blk];
                    }
                }
            }
        }
        if (column < elements_per_token) {
            if (use_low_precision) {
                #pragma unroll
                for (int word = 0; word < 24; word++) {
                    uint16_t _cvt_satfinite_e4m3x2_bf16x2_3;
                    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_3) : "r"(route_carriers[word]));
                    #else
                    {
                    const uint32_t _bf16x2 = (uint32_t)(route_carriers[word]);
                    const float _lo = __uint_as_float(_bf16x2 << 16);
                    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_3) : "f"(_hi), "f"(_lo));
                    }
                    #endif
                    uint32_t _cvt_bf16x2_e4m3x2_2 = cvt_bf16x2_e4m3x2(_cvt_satfinite_e4m3x2_bf16x2_3);
                    route_carriers[word] = _cvt_bf16x2_e4m3x2_2;
                }
            }
            float pair_values[16];
            #pragma unroll
            for (int _pair = 0; _pair < 8; _pair++) {
                (&pair_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(route_carriers[_pair]) << 16);
                (&pair_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(route_carriers[_pair]) & 0xffff0000u);
            }
            float accumulator[8];
            #pragma unroll
            for (int element = 0; element < 8; element++) {
                accumulator[element] = pair_values[element] + pair_values[8 + element];
            }
            #pragma unroll
            for (int _pair = 0; _pair < 8; _pair++) {
                (&pair_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(route_carriers[8 + _pair]) << 16);
                (&pair_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(route_carriers[8 + _pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int element_0 = 0; element_0 < 8; element_0++) {
                accumulator[element_0] = accumulator[element_0] + (pair_values[element_0] + pair_values[8 + element_0]);
            }
            #pragma unroll
            for (int _pair = 0; _pair < 8; _pair++) {
                (&pair_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(route_carriers[16 + _pair]) << 16);
                (&pair_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(route_carriers[16 + _pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int element_1 = 0; element_1 < 8; element_1++) {
                accumulator[element_1] = accumulator[element_1] + (pair_values[element_1] + pair_values[8 + element_1]);
            }
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accumulator[_lp*2 + 0], accumulator[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accumulator[_lp*2 + 2], accumulator[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accumulator[_lp*2 + 4], accumulator[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accumulator[_lp*2 + 6], accumulator[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_4 = output_bf16 + ((unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column);
                __builtin_assume(__isGlobal(_gst_4));
                reinterpret_cast<int4*>(_gst_4)[0] = reinterpret_cast<int4*>(_packed_0)[0];
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_topk8(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    __nv_bfloat16* workspace_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
    __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * 8ull;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * 8ull;
        #pragma unroll 1
        for (int column = tid * 8; column < elements_per_token; column += 2048) {
            float contributions[64];
            contributions[0] = 0.0f;
            contributions[1] = 0.0f;
            contributions[2] = 0.0f;
            contributions[3] = 0.0f;
            contributions[4] = 0.0f;
            contributions[5] = 0.0f;
            contributions[6] = 0.0f;
            contributions[7] = 0.0f;
            contributions[8] = 0.0f;
            contributions[9] = 0.0f;
            contributions[10] = 0.0f;
            contributions[11] = 0.0f;
            contributions[12] = 0.0f;
            contributions[13] = 0.0f;
            contributions[14] = 0.0f;
            contributions[15] = 0.0f;
            contributions[16] = 0.0f;
            contributions[17] = 0.0f;
            contributions[18] = 0.0f;
            contributions[19] = 0.0f;
            contributions[20] = 0.0f;
            contributions[21] = 0.0f;
            contributions[22] = 0.0f;
            contributions[23] = 0.0f;
            contributions[24] = 0.0f;
            contributions[25] = 0.0f;
            contributions[26] = 0.0f;
            contributions[27] = 0.0f;
            contributions[28] = 0.0f;
            contributions[29] = 0.0f;
            contributions[30] = 0.0f;
            contributions[31] = 0.0f;
            contributions[32] = 0.0f;
            contributions[33] = 0.0f;
            contributions[34] = 0.0f;
            contributions[35] = 0.0f;
            contributions[36] = 0.0f;
            contributions[37] = 0.0f;
            contributions[38] = 0.0f;
            contributions[39] = 0.0f;
            contributions[40] = 0.0f;
            contributions[41] = 0.0f;
            contributions[42] = 0.0f;
            contributions[43] = 0.0f;
            contributions[44] = 0.0f;
            contributions[45] = 0.0f;
            contributions[46] = 0.0f;
            contributions[47] = 0.0f;
            contributions[48] = 0.0f;
            contributions[49] = 0.0f;
            contributions[50] = 0.0f;
            contributions[51] = 0.0f;
            contributions[52] = 0.0f;
            contributions[53] = 0.0f;
            contributions[54] = 0.0f;
            contributions[55] = 0.0f;
            contributions[56] = 0.0f;
            contributions[57] = 0.0f;
            contributions[58] = 0.0f;
            contributions[59] = 0.0f;
            contributions[60] = 0.0f;
            contributions[61] = 0.0f;
            contributions[62] = 0.0f;
            contributions[63] = 0.0f;
            #pragma unroll
            for (int route = 0; route < 8; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    if (use_low_precision) {
                        unsigned int route_carriers[4];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(workspace_bf16 + byte_index / 2ull);
                            uint4* _vdst_0 = reinterpret_cast<uint4*>(&route_carriers[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_0[_blk] = _vptr_0[_blk];
                            }
                        }
                        #pragma unroll
                        for (int word = 0; word < 4; word++) {
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_2;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_2) : "r"(route_carriers[word]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(route_carriers[word]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_2) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _cvt_bf16x2_e4m3x2_1 = cvt_bf16x2_e4m3x2(_cvt_satfinite_e4m3x2_bf16x2_2);
                            route_carriers[word] = _cvt_bf16x2_e4m3x2_1;
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&contributions[route * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(route_carriers[_pair]) << 16);
                            (&contributions[route * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(route_carriers[_pair]) & 0xffff0000u);
                        }
                    } else {
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(workspace_bf16 + byte_index / 2ull);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        #pragma unroll
                        for (int element = 0; element < 8; element++) {
                            contributions[route * 8 + element] = _vec_load_0[element];
                        }
                    }
                }
            }
            #pragma unroll
            for (int element_1 = 0; element_1 < 8; element_1++) {
                contributions[element_1] = contributions[element_1] + contributions[8 + element_1];
                contributions[16 + element_1] = contributions[16 + element_1] + contributions[24 + element_1];
                contributions[32 + element_1] = contributions[32 + element_1] + contributions[40 + element_1];
                contributions[48 + element_1] = contributions[48 + element_1] + contributions[56 + element_1];
                contributions[element_1] = contributions[element_1] + contributions[16 + element_1];
                contributions[32 + element_1] = contributions[32 + element_1] + contributions[48 + element_1];
                contributions[element_1] = contributions[element_1] + contributions[32 + element_1];
            }
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 0], contributions[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 2], contributions[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 4], contributions[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 6], contributions[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_4 = output_bf16 + ((unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column);
                __builtin_assume(__isGlobal(_gst_4));
                reinterpret_cast<int4*>(_gst_4)[0] = reinterpret_cast<int4*>(_packed_0)[0];
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#define TOP_K 1

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_vector_top_k_1(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    __nv_bfloat16* workspace_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
    __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[1];
        int send_indices[1];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid * 8; column < elements_per_token; column += 2048) {
            float contributions[8];
            contributions[0] = 0.0f;
            contributions[1] = 0.0f;
            contributions[2] = 0.0f;
            contributions[3] = 0.0f;
            contributions[4] = 0.0f;
            contributions[5] = 0.0f;
            contributions[6] = 0.0f;
            contributions[7] = 0.0f;
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    if (use_low_precision) {
                        unsigned int route_carriers[4];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(workspace_bf16 + byte_index / 2ull);
                            uint4* _vdst_0 = reinterpret_cast<uint4*>(&route_carriers[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_0[_blk] = _vptr_0[_blk];
                            }
                        }
                        #pragma unroll
                        for (int word = 0; word < 4; word++) {
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_2;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_2) : "r"(route_carriers[word]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(route_carriers[word]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_2) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _cvt_bf16x2_e4m3x2_1 = cvt_bf16x2_e4m3x2(_cvt_satfinite_e4m3x2_bf16x2_2);
                            route_carriers[word] = _cvt_bf16x2_e4m3x2_1;
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&contributions[route * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(route_carriers[_pair]) << 16);
                            (&contributions[route * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(route_carriers[_pair]) & 0xffff0000u);
                        }
                    } else {
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(workspace_bf16 + byte_index / 2ull);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        #pragma unroll
                        for (int element = 0; element < 8; element++) {
                            contributions[route * 8 + element] = _vec_load_0[element];
                        }
                    }
                }
            }
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 0], contributions[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 2], contributions[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 4], contributions[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 6], contributions[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_4 = output_bf16 + ((unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column);
                __builtin_assume(__isGlobal(_gst_4));
                reinterpret_cast<int4*>(_gst_4)[0] = reinterpret_cast<int4*>(_packed_0)[0];
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 2

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_vector_top_k_2(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    __nv_bfloat16* workspace_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
    __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[2];
        int send_indices[2];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid * 8; column < elements_per_token; column += 2048) {
            float contributions[16];
            contributions[0] = 0.0f;
            contributions[1] = 0.0f;
            contributions[2] = 0.0f;
            contributions[3] = 0.0f;
            contributions[4] = 0.0f;
            contributions[5] = 0.0f;
            contributions[6] = 0.0f;
            contributions[7] = 0.0f;
            contributions[8] = 0.0f;
            contributions[9] = 0.0f;
            contributions[10] = 0.0f;
            contributions[11] = 0.0f;
            contributions[12] = 0.0f;
            contributions[13] = 0.0f;
            contributions[14] = 0.0f;
            contributions[15] = 0.0f;
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    if (use_low_precision) {
                        unsigned int route_carriers[4];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(workspace_bf16 + byte_index / 2ull);
                            uint4* _vdst_0 = reinterpret_cast<uint4*>(&route_carriers[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_0[_blk] = _vptr_0[_blk];
                            }
                        }
                        #pragma unroll
                        for (int word = 0; word < 4; word++) {
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_2;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_2) : "r"(route_carriers[word]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(route_carriers[word]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_2) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _cvt_bf16x2_e4m3x2_1 = cvt_bf16x2_e4m3x2(_cvt_satfinite_e4m3x2_bf16x2_2);
                            route_carriers[word] = _cvt_bf16x2_e4m3x2_1;
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&contributions[route * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(route_carriers[_pair]) << 16);
                            (&contributions[route * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(route_carriers[_pair]) & 0xffff0000u);
                        }
                    } else {
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(workspace_bf16 + byte_index / 2ull);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        #pragma unroll
                        for (int element = 0; element < 8; element++) {
                            contributions[route * 8 + element] = _vec_load_0[element];
                        }
                    }
                }
            }
            #pragma unroll
            for (int element_1 = 0; element_1 < 8; element_1++) {
                contributions[element_1] = contributions[element_1] + contributions[8 + element_1];
            }
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 0], contributions[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 2], contributions[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 4], contributions[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 6], contributions[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_4 = output_bf16 + ((unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column);
                __builtin_assume(__isGlobal(_gst_4));
                reinterpret_cast<int4*>(_gst_4)[0] = reinterpret_cast<int4*>(_packed_0)[0];
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 4

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_vector_top_k_4(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    __nv_bfloat16* workspace_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);
    __nv_bfloat16* output_bf16 = reinterpret_cast<__nv_bfloat16*>(output);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[4];
        int send_indices[4];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column = tid * 8; column < elements_per_token; column += 2048) {
            float contributions[32];
            contributions[0] = 0.0f;
            contributions[1] = 0.0f;
            contributions[2] = 0.0f;
            contributions[3] = 0.0f;
            contributions[4] = 0.0f;
            contributions[5] = 0.0f;
            contributions[6] = 0.0f;
            contributions[7] = 0.0f;
            contributions[8] = 0.0f;
            contributions[9] = 0.0f;
            contributions[10] = 0.0f;
            contributions[11] = 0.0f;
            contributions[12] = 0.0f;
            contributions[13] = 0.0f;
            contributions[14] = 0.0f;
            contributions[15] = 0.0f;
            contributions[16] = 0.0f;
            contributions[17] = 0.0f;
            contributions[18] = 0.0f;
            contributions[19] = 0.0f;
            contributions[20] = 0.0f;
            contributions[21] = 0.0f;
            contributions[22] = 0.0f;
            contributions[23] = 0.0f;
            contributions[24] = 0.0f;
            contributions[25] = 0.0f;
            contributions[26] = 0.0f;
            contributions[27] = 0.0f;
            contributions[28] = 0.0f;
            contributions[29] = 0.0f;
            contributions[30] = 0.0f;
            contributions[31] = 0.0f;
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = target_ranks[route];
                int send_index = send_indices[route];
                if (target_rank >= 0 && send_index >= 0) {
                    unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                    unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                    if (use_low_precision) {
                        unsigned int route_carriers[4];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(workspace_bf16 + byte_index / 2ull);
                            uint4* _vdst_0 = reinterpret_cast<uint4*>(&route_carriers[0]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_0[_blk] = _vptr_0[_blk];
                            }
                        }
                        #pragma unroll
                        for (int word = 0; word < 4; word++) {
                            uint16_t _cvt_satfinite_e4m3x2_bf16x2_2;
                            #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                            asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_2) : "r"(route_carriers[word]));
                            #else
                            {
                            const uint32_t _bf16x2 = (uint32_t)(route_carriers[word]);
                            const float _lo = __uint_as_float(_bf16x2 << 16);
                            const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_2) : "f"(_hi), "f"(_lo));
                            }
                            #endif
                            uint32_t _cvt_bf16x2_e4m3x2_1 = cvt_bf16x2_e4m3x2(_cvt_satfinite_e4m3x2_bf16x2_2);
                            route_carriers[word] = _cvt_bf16x2_e4m3x2_1;
                        }
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&contributions[route * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(route_carriers[_pair]) << 16);
                            (&contributions[route * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(route_carriers[_pair]) & 0xffff0000u);
                        }
                    } else {
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(workspace_bf16 + byte_index / 2ull);
                            uint4 _vld_3[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                }
                            }
                        }
                        #pragma unroll
                        for (int element = 0; element < 8; element++) {
                            contributions[route * 8 + element] = _vec_load_0[element];
                        }
                    }
                }
            }
            #pragma unroll
            for (int element_1 = 0; element_1 < 8; element_1++) {
                contributions[element_1] = contributions[element_1] + contributions[8 + element_1];
                contributions[16 + element_1] = contributions[16 + element_1] + contributions[24 + element_1];
                contributions[element_1] = contributions[element_1] + contributions[16 + element_1];
            }
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 0], contributions[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 2], contributions[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 4], contributions[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(contributions[_lp*2 + 6], contributions[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            {
                auto _gst_4 = output_bf16 + ((unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column);
                __builtin_assume(__isGlobal(_gst_4));
                reinterpret_cast<int4*>(_gst_4)[0] = reinterpret_cast<int4*>(_packed_0)[0];
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 1

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_1(uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    uint8_t* workspace_fp8 = reinterpret_cast<uint8_t*>(workspace);
    float* workspace_f32 = reinterpret_cast<float*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[1];
        int send_indices[1];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column_base = 0; column_base < elements_per_token; column_base += 512) {
            int column = column_base + 2 * tid;
            int active = 0;
            float carrier_values[2];
            carrier_values[0] = 0.0f;
            carrier_values[1] = 0.0f;
            if (column < elements_per_token) {
                active = 1;
                float contributions[2];
                contributions[0] = 0.0f;
                contributions[1] = 0.0f;
                #pragma unroll
                for (int route = 0; route < TOP_K; route++) {
                    int target_rank = target_ranks[route];
                    int send_index = send_indices[route];
                    if (target_rank >= 0 && send_index >= 0) {
                        unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                        unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                        if (payload_dtype_code == 0 || payload_dtype_code == 1) {
                            unsigned int pair_word[1];
                            {
                                pair_word[0] = *reinterpret_cast<const unsigned int*>(workspace_u32 + (byte_index / 4ull));
                            }
                            if (payload_dtype_code == 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    (&contributions[2 * route + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) << 16);
                                    (&contributions[2 * route + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) & 0xffff0000u);
                                }
                            } else {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 h_lo, h_hi;\n\t"
                                        ".reg .b32 f_lo, f_hi;\n\t"
                                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&contributions[2 * route + _pair * 2]))
                                        : "r"(pair_word[_pair]));
                                }
                            }
                        }
                        if (payload_dtype_code == 2) {
                            {
                                uint16_t _e4m3x2_0 = *reinterpret_cast<const uint16_t*>(workspace_fp8 + byte_index);
                                uint32_t _f16x2_0;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                                uint16_t _h0_0 = (uint16_t)((_f16x2_0 >> 0) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 0]) : "h"(_h0_0));
                                uint16_t _h1_0 = (uint16_t)((_f16x2_0 >> 16) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 1]) : "h"(_h1_0));
                            }
                        }
                        if (payload_dtype_code == 3) {
                            {
                                float2 _v2_1 = *reinterpret_cast<const float2*>(workspace_f32 + byte_index / 4ull);
                                contributions[2 * route] = _v2_1.x;
                                contributions[2 * route + 1] = _v2_1.y;
                            }
                        }
                        if (use_low_precision) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_2) : "f"(contributions[2 * route + 1]), "f"(contributions[2 * route]));
                            float2 rounded;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(rounded.x), "=f"(rounded.y) : "h"(_cvt_satfinite_e4m3x2_f32_2));
                            contributions[2 * route] = rounded.x;
                            contributions[2 * route + 1] = rounded.y;
                        }
                    }
                }
                float result_0 = contributions[0];
                if (payload_dtype_code == 1) {
                    carrier_values[0] = (float)(__half)result_0;
                } else {
                    carrier_values[0] = (float)(__nv_bfloat16)result_0;
                }
                float result_1 = contributions[1];
                if (payload_dtype_code == 1) {
                    carrier_values[1] = (float)(__half)result_1;
                } else {
                    carrier_values[1] = (float)(__nv_bfloat16)result_1;
                }
            }
            int block_column = column / block_size;
            float block_max;
            asm("max.f32 %0, %1, %2;" : "=f"(block_max) : "f"(fabsf(carrier_values[0])), "f"(fabsf(carrier_values[1])));
            if (quant_mode == 3) {
                block_max = max_noftz(block_max, 0.0f);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_5);
            } else {
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
                block_max = max_noftz(block_max, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_8);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_9);
            }
            int scale_byte = 0;
            float actual_scale = 0.0f;
            float fp8_scale = 0.0f;
            if (quant_mode == 3) {
                float reciprocal_six = approx_rcp(6.0f);
                float sf_value = output_scalar_scale * (block_max * reciprocal_six);
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
                uint32_t _f16x2_decode_13;
                float _fp8_e4m3_decode_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
                fp8_scale = _fp8_e4m3_decode_10;
                if (block_max != 0.0f) {
                    float reciprocal_scalar = approx_rcp(output_scalar_scale);
                    float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
                    actual_scale = _rcp_14;
                }
            } else {
                float denominator = 6.0f;
                if (quant_mode == 1) {
                    denominator = 448.0f;
                }
                float _rcp_15 = approx_rcp(denominator);
                float raw_scale = block_max * _rcp_15;
                int raw_bits = 0;
                raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
                int exponent = raw_bits >> 23 & 255;
                int mantissa = raw_bits & 8388607;
                int round_up = 0;
                if (mantissa != 0) {
                    if (exponent != 0) {
                        round_up = 1;
                    } else if (mantissa > 4194304) {
                        round_up = 1;
                    }
                }
                if (raw_scale > 0.0f) {
                    scale_byte = exponent + round_up;
                    if (scale_byte > 254) {
                        scale_byte = 254;
                    }
                }
                if (quant_mode == 1) {
                    int scale_bits = scale_byte << 23;
                    if (scale_byte == 0) {
                        scale_bits = 4194304;
                    }
                    float decoded_scale = 0.0f;
                    decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
                    float _rcp_16 = approx_rcp(decoded_scale);
                    actual_scale = _rcp_16;
                } else if (block_max != 0.0f) {
                    if (scale_byte == 0) {
                        actual_scale = 1.0f;
                    } else {
                        float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                        actual_scale = _exp2_17;
                    }
                }
            }
            int scale_index = token * blocks_per_row + block_column;
            if (scale_layout == 0) {
                scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
            }
            if (scale_layout == 1) {
                int tiles = padded_scale_cols / 4;
                scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
            }
            int block_lane = lane & 15;
            if (quant_mode == 3) {
                block_lane = lane & 7;
            }
            if (active != 0) {
                if (block_lane == 0) {
                    if (quant_mode == 3) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                        *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
                    } else {
                        scales_u8[scale_index] = scale_byte;
                    }
                }
                float normalized_low = carrier_values[0] * actual_scale;
                float normalized_high = carrier_values[1] * actual_scale;
                if (quant_mode == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(normalized_high), "f"(normalized_low));
                    *reinterpret_cast<unsigned short*>(quantized_fp8 + (token * elements_per_token + column)) = (unsigned short)(_cvt_satfinite_e4m3x2_f32_19);
                } else {
                    uint32_t _fp4_e2m1_pack_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_e2m1_pack_pair_20) : "f"(normalized_low), "f"(normalized_high));
                    quantized_packed[token * (elements_per_token / 2) + column / 2] = _fp4_e2m1_pack_pair_20;
                }
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 2

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_2(uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    uint8_t* workspace_fp8 = reinterpret_cast<uint8_t*>(workspace);
    float* workspace_f32 = reinterpret_cast<float*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[2];
        int send_indices[2];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column_base = 0; column_base < elements_per_token; column_base += 512) {
            int column = column_base + 2 * tid;
            int active = 0;
            float carrier_values[2];
            carrier_values[0] = 0.0f;
            carrier_values[1] = 0.0f;
            if (column < elements_per_token) {
                active = 1;
                float contributions[4];
                contributions[0] = 0.0f;
                contributions[1] = 0.0f;
                contributions[2] = 0.0f;
                contributions[3] = 0.0f;
                #pragma unroll
                for (int route = 0; route < TOP_K; route++) {
                    int target_rank = target_ranks[route];
                    int send_index = send_indices[route];
                    if (target_rank >= 0 && send_index >= 0) {
                        unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                        unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                        if (payload_dtype_code == 0 || payload_dtype_code == 1) {
                            unsigned int pair_word[1];
                            {
                                pair_word[0] = *reinterpret_cast<const unsigned int*>(workspace_u32 + (byte_index / 4ull));
                            }
                            if (payload_dtype_code == 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    (&contributions[2 * route + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) << 16);
                                    (&contributions[2 * route + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) & 0xffff0000u);
                                }
                            } else {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 h_lo, h_hi;\n\t"
                                        ".reg .b32 f_lo, f_hi;\n\t"
                                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&contributions[2 * route + _pair * 2]))
                                        : "r"(pair_word[_pair]));
                                }
                            }
                        }
                        if (payload_dtype_code == 2) {
                            {
                                uint16_t _e4m3x2_0 = *reinterpret_cast<const uint16_t*>(workspace_fp8 + byte_index);
                                uint32_t _f16x2_0;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                                uint16_t _h0_0 = (uint16_t)((_f16x2_0 >> 0) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 0]) : "h"(_h0_0));
                                uint16_t _h1_0 = (uint16_t)((_f16x2_0 >> 16) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 1]) : "h"(_h1_0));
                            }
                        }
                        if (payload_dtype_code == 3) {
                            {
                                float2 _v2_1 = *reinterpret_cast<const float2*>(workspace_f32 + byte_index / 4ull);
                                contributions[2 * route] = _v2_1.x;
                                contributions[2 * route + 1] = _v2_1.y;
                            }
                        }
                        if (use_low_precision) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_2) : "f"(contributions[2 * route + 1]), "f"(contributions[2 * route]));
                            float2 rounded;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(rounded.x), "=f"(rounded.y) : "h"(_cvt_satfinite_e4m3x2_f32_2));
                            contributions[2 * route] = rounded.x;
                            contributions[2 * route + 1] = rounded.y;
                        }
                    }
                }
                contributions[0] = contributions[0] + contributions[2];
                float result_0 = contributions[0];
                if (payload_dtype_code == 1) {
                    carrier_values[0] = (float)(__half)result_0;
                } else {
                    carrier_values[0] = (float)(__nv_bfloat16)result_0;
                }
                contributions[1] = contributions[1] + contributions[3];
                float result_1 = contributions[1];
                if (payload_dtype_code == 1) {
                    carrier_values[1] = (float)(__half)result_1;
                } else {
                    carrier_values[1] = (float)(__nv_bfloat16)result_1;
                }
            }
            int block_column = column / block_size;
            float block_max;
            asm("max.f32 %0, %1, %2;" : "=f"(block_max) : "f"(fabsf(carrier_values[0])), "f"(fabsf(carrier_values[1])));
            if (quant_mode == 3) {
                block_max = max_noftz(block_max, 0.0f);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_5);
            } else {
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
                block_max = max_noftz(block_max, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_8);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_9);
            }
            int scale_byte = 0;
            float actual_scale = 0.0f;
            float fp8_scale = 0.0f;
            if (quant_mode == 3) {
                float reciprocal_six = approx_rcp(6.0f);
                float sf_value = output_scalar_scale * (block_max * reciprocal_six);
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
                uint32_t _f16x2_decode_13;
                float _fp8_e4m3_decode_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
                fp8_scale = _fp8_e4m3_decode_10;
                if (block_max != 0.0f) {
                    float reciprocal_scalar = approx_rcp(output_scalar_scale);
                    float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
                    actual_scale = _rcp_14;
                }
            } else {
                float denominator = 6.0f;
                if (quant_mode == 1) {
                    denominator = 448.0f;
                }
                float _rcp_15 = approx_rcp(denominator);
                float raw_scale = block_max * _rcp_15;
                int raw_bits = 0;
                raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
                int exponent = raw_bits >> 23 & 255;
                int mantissa = raw_bits & 8388607;
                int round_up = 0;
                if (mantissa != 0) {
                    if (exponent != 0) {
                        round_up = 1;
                    } else if (mantissa > 4194304) {
                        round_up = 1;
                    }
                }
                if (raw_scale > 0.0f) {
                    scale_byte = exponent + round_up;
                    if (scale_byte > 254) {
                        scale_byte = 254;
                    }
                }
                if (quant_mode == 1) {
                    int scale_bits = scale_byte << 23;
                    if (scale_byte == 0) {
                        scale_bits = 4194304;
                    }
                    float decoded_scale = 0.0f;
                    decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
                    float _rcp_16 = approx_rcp(decoded_scale);
                    actual_scale = _rcp_16;
                } else if (block_max != 0.0f) {
                    if (scale_byte == 0) {
                        actual_scale = 1.0f;
                    } else {
                        float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                        actual_scale = _exp2_17;
                    }
                }
            }
            int scale_index = token * blocks_per_row + block_column;
            if (scale_layout == 0) {
                scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
            }
            if (scale_layout == 1) {
                int tiles = padded_scale_cols / 4;
                scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
            }
            int block_lane = lane & 15;
            if (quant_mode == 3) {
                block_lane = lane & 7;
            }
            if (active != 0) {
                if (block_lane == 0) {
                    if (quant_mode == 3) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                        *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
                    } else {
                        scales_u8[scale_index] = scale_byte;
                    }
                }
                float normalized_low = carrier_values[0] * actual_scale;
                float normalized_high = carrier_values[1] * actual_scale;
                if (quant_mode == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(normalized_high), "f"(normalized_low));
                    *reinterpret_cast<unsigned short*>(quantized_fp8 + (token * elements_per_token + column)) = (unsigned short)(_cvt_satfinite_e4m3x2_f32_19);
                } else {
                    uint32_t _fp4_e2m1_pack_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_e2m1_pack_pair_20) : "f"(normalized_low), "f"(normalized_high));
                    quantized_packed[token * (elements_per_token / 2) + column / 2] = _fp4_e2m1_pack_pair_20;
                }
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 4

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_4(uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    uint8_t* workspace_fp8 = reinterpret_cast<uint8_t*>(workspace);
    float* workspace_f32 = reinterpret_cast<float*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[4];
        int send_indices[4];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column_base = 0; column_base < elements_per_token; column_base += 512) {
            int column = column_base + 2 * tid;
            int active = 0;
            float carrier_values[2];
            carrier_values[0] = 0.0f;
            carrier_values[1] = 0.0f;
            if (column < elements_per_token) {
                active = 1;
                float contributions[8];
                contributions[0] = 0.0f;
                contributions[1] = 0.0f;
                contributions[2] = 0.0f;
                contributions[3] = 0.0f;
                contributions[4] = 0.0f;
                contributions[5] = 0.0f;
                contributions[6] = 0.0f;
                contributions[7] = 0.0f;
                #pragma unroll
                for (int route = 0; route < TOP_K; route++) {
                    int target_rank = target_ranks[route];
                    int send_index = send_indices[route];
                    if (target_rank >= 0 && send_index >= 0) {
                        unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                        unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                        if (payload_dtype_code == 0 || payload_dtype_code == 1) {
                            unsigned int pair_word[1];
                            {
                                pair_word[0] = *reinterpret_cast<const unsigned int*>(workspace_u32 + (byte_index / 4ull));
                            }
                            if (payload_dtype_code == 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    (&contributions[2 * route + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) << 16);
                                    (&contributions[2 * route + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) & 0xffff0000u);
                                }
                            } else {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 h_lo, h_hi;\n\t"
                                        ".reg .b32 f_lo, f_hi;\n\t"
                                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&contributions[2 * route + _pair * 2]))
                                        : "r"(pair_word[_pair]));
                                }
                            }
                        }
                        if (payload_dtype_code == 2) {
                            {
                                uint16_t _e4m3x2_0 = *reinterpret_cast<const uint16_t*>(workspace_fp8 + byte_index);
                                uint32_t _f16x2_0;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                                uint16_t _h0_0 = (uint16_t)((_f16x2_0 >> 0) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 0]) : "h"(_h0_0));
                                uint16_t _h1_0 = (uint16_t)((_f16x2_0 >> 16) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 1]) : "h"(_h1_0));
                            }
                        }
                        if (payload_dtype_code == 3) {
                            {
                                float2 _v2_1 = *reinterpret_cast<const float2*>(workspace_f32 + byte_index / 4ull);
                                contributions[2 * route] = _v2_1.x;
                                contributions[2 * route + 1] = _v2_1.y;
                            }
                        }
                        if (use_low_precision) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_2) : "f"(contributions[2 * route + 1]), "f"(contributions[2 * route]));
                            float2 rounded;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(rounded.x), "=f"(rounded.y) : "h"(_cvt_satfinite_e4m3x2_f32_2));
                            contributions[2 * route] = rounded.x;
                            contributions[2 * route + 1] = rounded.y;
                        }
                    }
                }
                contributions[0] = contributions[0] + contributions[2];
                contributions[4] = contributions[4] + contributions[6];
                contributions[0] = contributions[0] + contributions[4];
                float result_0 = contributions[0];
                if (payload_dtype_code == 1) {
                    carrier_values[0] = (float)(__half)result_0;
                } else {
                    carrier_values[0] = (float)(__nv_bfloat16)result_0;
                }
                contributions[1] = contributions[1] + contributions[3];
                contributions[5] = contributions[5] + contributions[7];
                contributions[1] = contributions[1] + contributions[5];
                float result_1 = contributions[1];
                if (payload_dtype_code == 1) {
                    carrier_values[1] = (float)(__half)result_1;
                } else {
                    carrier_values[1] = (float)(__nv_bfloat16)result_1;
                }
            }
            int block_column = column / block_size;
            float block_max;
            asm("max.f32 %0, %1, %2;" : "=f"(block_max) : "f"(fabsf(carrier_values[0])), "f"(fabsf(carrier_values[1])));
            if (quant_mode == 3) {
                block_max = max_noftz(block_max, 0.0f);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_5);
            } else {
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
                block_max = max_noftz(block_max, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_8);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_9);
            }
            int scale_byte = 0;
            float actual_scale = 0.0f;
            float fp8_scale = 0.0f;
            if (quant_mode == 3) {
                float reciprocal_six = approx_rcp(6.0f);
                float sf_value = output_scalar_scale * (block_max * reciprocal_six);
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
                uint32_t _f16x2_decode_13;
                float _fp8_e4m3_decode_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
                fp8_scale = _fp8_e4m3_decode_10;
                if (block_max != 0.0f) {
                    float reciprocal_scalar = approx_rcp(output_scalar_scale);
                    float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
                    actual_scale = _rcp_14;
                }
            } else {
                float denominator = 6.0f;
                if (quant_mode == 1) {
                    denominator = 448.0f;
                }
                float _rcp_15 = approx_rcp(denominator);
                float raw_scale = block_max * _rcp_15;
                int raw_bits = 0;
                raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
                int exponent = raw_bits >> 23 & 255;
                int mantissa = raw_bits & 8388607;
                int round_up = 0;
                if (mantissa != 0) {
                    if (exponent != 0) {
                        round_up = 1;
                    } else if (mantissa > 4194304) {
                        round_up = 1;
                    }
                }
                if (raw_scale > 0.0f) {
                    scale_byte = exponent + round_up;
                    if (scale_byte > 254) {
                        scale_byte = 254;
                    }
                }
                if (quant_mode == 1) {
                    int scale_bits = scale_byte << 23;
                    if (scale_byte == 0) {
                        scale_bits = 4194304;
                    }
                    float decoded_scale = 0.0f;
                    decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
                    float _rcp_16 = approx_rcp(decoded_scale);
                    actual_scale = _rcp_16;
                } else if (block_max != 0.0f) {
                    if (scale_byte == 0) {
                        actual_scale = 1.0f;
                    } else {
                        float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                        actual_scale = _exp2_17;
                    }
                }
            }
            int scale_index = token * blocks_per_row + block_column;
            if (scale_layout == 0) {
                scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
            }
            if (scale_layout == 1) {
                int tiles = padded_scale_cols / 4;
                scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
            }
            int block_lane = lane & 15;
            if (quant_mode == 3) {
                block_lane = lane & 7;
            }
            if (active != 0) {
                if (block_lane == 0) {
                    if (quant_mode == 3) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                        *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
                    } else {
                        scales_u8[scale_index] = scale_byte;
                    }
                }
                float normalized_low = carrier_values[0] * actual_scale;
                float normalized_high = carrier_values[1] * actual_scale;
                if (quant_mode == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(normalized_high), "f"(normalized_low));
                    *reinterpret_cast<unsigned short*>(quantized_fp8 + (token * elements_per_token + column)) = (unsigned short)(_cvt_satfinite_e4m3x2_f32_19);
                } else {
                    uint32_t _fp4_e2m1_pack_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_e2m1_pack_pair_20) : "f"(normalized_low), "f"(normalized_high));
                    quantized_packed[token * (elements_per_token / 2) + column / 2] = _fp4_e2m1_pack_pair_20;
                }
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 6

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_6(uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    uint8_t* workspace_fp8 = reinterpret_cast<uint8_t*>(workspace);
    float* workspace_f32 = reinterpret_cast<float*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[6];
        int send_indices[6];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column_base = 0; column_base < elements_per_token; column_base += 512) {
            int column = column_base + 2 * tid;
            int active = 0;
            float carrier_values[2];
            carrier_values[0] = 0.0f;
            carrier_values[1] = 0.0f;
            if (column < elements_per_token) {
                active = 1;
                float contributions[12];
                contributions[0] = 0.0f;
                contributions[1] = 0.0f;
                contributions[2] = 0.0f;
                contributions[3] = 0.0f;
                contributions[4] = 0.0f;
                contributions[5] = 0.0f;
                contributions[6] = 0.0f;
                contributions[7] = 0.0f;
                contributions[8] = 0.0f;
                contributions[9] = 0.0f;
                contributions[10] = 0.0f;
                contributions[11] = 0.0f;
                #pragma unroll
                for (int route = 0; route < TOP_K; route++) {
                    int target_rank = target_ranks[route];
                    int send_index = send_indices[route];
                    if (target_rank >= 0 && send_index >= 0) {
                        unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                        unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                        if (payload_dtype_code == 0 || payload_dtype_code == 1) {
                            unsigned int pair_word[1];
                            {
                                pair_word[0] = *reinterpret_cast<const unsigned int*>(workspace_u32 + (byte_index / 4ull));
                            }
                            if (payload_dtype_code == 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    (&contributions[2 * route + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) << 16);
                                    (&contributions[2 * route + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) & 0xffff0000u);
                                }
                            } else {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 h_lo, h_hi;\n\t"
                                        ".reg .b32 f_lo, f_hi;\n\t"
                                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&contributions[2 * route + _pair * 2]))
                                        : "r"(pair_word[_pair]));
                                }
                            }
                        }
                        if (payload_dtype_code == 2) {
                            {
                                uint16_t _e4m3x2_0 = *reinterpret_cast<const uint16_t*>(workspace_fp8 + byte_index);
                                uint32_t _f16x2_0;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                                uint16_t _h0_0 = (uint16_t)((_f16x2_0 >> 0) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 0]) : "h"(_h0_0));
                                uint16_t _h1_0 = (uint16_t)((_f16x2_0 >> 16) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 1]) : "h"(_h1_0));
                            }
                        }
                        if (payload_dtype_code == 3) {
                            {
                                float2 _v2_1 = *reinterpret_cast<const float2*>(workspace_f32 + byte_index / 4ull);
                                contributions[2 * route] = _v2_1.x;
                                contributions[2 * route + 1] = _v2_1.y;
                            }
                        }
                        if (use_low_precision) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_2) : "f"(contributions[2 * route + 1]), "f"(contributions[2 * route]));
                            float2 rounded;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(rounded.x), "=f"(rounded.y) : "h"(_cvt_satfinite_e4m3x2_f32_2));
                            contributions[2 * route] = rounded.x;
                            contributions[2 * route + 1] = rounded.y;
                        }
                    }
                }
                contributions[0] = contributions[0] + contributions[2];
                contributions[4] = contributions[4] + contributions[6];
                contributions[8] = contributions[8] + contributions[10];
                contributions[0] = contributions[0] + contributions[4];
                contributions[0] = contributions[0] + contributions[8];
                float result_0 = contributions[0];
                if (payload_dtype_code == 1) {
                    carrier_values[0] = (float)(__half)result_0;
                } else {
                    carrier_values[0] = (float)(__nv_bfloat16)result_0;
                }
                contributions[1] = contributions[1] + contributions[3];
                contributions[5] = contributions[5] + contributions[7];
                contributions[9] = contributions[9] + contributions[11];
                contributions[1] = contributions[1] + contributions[5];
                contributions[1] = contributions[1] + contributions[9];
                float result_1 = contributions[1];
                if (payload_dtype_code == 1) {
                    carrier_values[1] = (float)(__half)result_1;
                } else {
                    carrier_values[1] = (float)(__nv_bfloat16)result_1;
                }
            }
            int block_column = column / block_size;
            float block_max;
            asm("max.f32 %0, %1, %2;" : "=f"(block_max) : "f"(fabsf(carrier_values[0])), "f"(fabsf(carrier_values[1])));
            if (quant_mode == 3) {
                block_max = max_noftz(block_max, 0.0f);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_5);
            } else {
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
                block_max = max_noftz(block_max, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_8);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_9);
            }
            int scale_byte = 0;
            float actual_scale = 0.0f;
            float fp8_scale = 0.0f;
            if (quant_mode == 3) {
                float reciprocal_six = approx_rcp(6.0f);
                float sf_value = output_scalar_scale * (block_max * reciprocal_six);
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
                uint32_t _f16x2_decode_13;
                float _fp8_e4m3_decode_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
                fp8_scale = _fp8_e4m3_decode_10;
                if (block_max != 0.0f) {
                    float reciprocal_scalar = approx_rcp(output_scalar_scale);
                    float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
                    actual_scale = _rcp_14;
                }
            } else {
                float denominator = 6.0f;
                if (quant_mode == 1) {
                    denominator = 448.0f;
                }
                float _rcp_15 = approx_rcp(denominator);
                float raw_scale = block_max * _rcp_15;
                int raw_bits = 0;
                raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
                int exponent = raw_bits >> 23 & 255;
                int mantissa = raw_bits & 8388607;
                int round_up = 0;
                if (mantissa != 0) {
                    if (exponent != 0) {
                        round_up = 1;
                    } else if (mantissa > 4194304) {
                        round_up = 1;
                    }
                }
                if (raw_scale > 0.0f) {
                    scale_byte = exponent + round_up;
                    if (scale_byte > 254) {
                        scale_byte = 254;
                    }
                }
                if (quant_mode == 1) {
                    int scale_bits = scale_byte << 23;
                    if (scale_byte == 0) {
                        scale_bits = 4194304;
                    }
                    float decoded_scale = 0.0f;
                    decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
                    float _rcp_16 = approx_rcp(decoded_scale);
                    actual_scale = _rcp_16;
                } else if (block_max != 0.0f) {
                    if (scale_byte == 0) {
                        actual_scale = 1.0f;
                    } else {
                        float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                        actual_scale = _exp2_17;
                    }
                }
            }
            int scale_index = token * blocks_per_row + block_column;
            if (scale_layout == 0) {
                scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
            }
            if (scale_layout == 1) {
                int tiles = padded_scale_cols / 4;
                scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
            }
            int block_lane = lane & 15;
            if (quant_mode == 3) {
                block_lane = lane & 7;
            }
            if (active != 0) {
                if (block_lane == 0) {
                    if (quant_mode == 3) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                        *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
                    } else {
                        scales_u8[scale_index] = scale_byte;
                    }
                }
                float normalized_low = carrier_values[0] * actual_scale;
                float normalized_high = carrier_values[1] * actual_scale;
                if (quant_mode == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(normalized_high), "f"(normalized_low));
                    *reinterpret_cast<unsigned short*>(quantized_fp8 + (token * elements_per_token + column)) = (unsigned short)(_cvt_satfinite_e4m3x2_f32_19);
                } else {
                    uint32_t _fp4_e2m1_pack_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_e2m1_pack_pair_20) : "f"(normalized_low), "f"(normalized_high));
                    quantized_packed[token * (elements_per_token / 2) + column / 2] = _fp4_e2m1_pack_pair_20;
                }
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 8

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_8(uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    uint8_t* workspace_fp8 = reinterpret_cast<uint8_t*>(workspace);
    float* workspace_f32 = reinterpret_cast<float*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[8];
        int send_indices[8];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column_base = 0; column_base < elements_per_token; column_base += 512) {
            int column = column_base + 2 * tid;
            int active = 0;
            float carrier_values[2];
            carrier_values[0] = 0.0f;
            carrier_values[1] = 0.0f;
            if (column < elements_per_token) {
                active = 1;
                float contributions[16];
                contributions[0] = 0.0f;
                contributions[1] = 0.0f;
                contributions[2] = 0.0f;
                contributions[3] = 0.0f;
                contributions[4] = 0.0f;
                contributions[5] = 0.0f;
                contributions[6] = 0.0f;
                contributions[7] = 0.0f;
                contributions[8] = 0.0f;
                contributions[9] = 0.0f;
                contributions[10] = 0.0f;
                contributions[11] = 0.0f;
                contributions[12] = 0.0f;
                contributions[13] = 0.0f;
                contributions[14] = 0.0f;
                contributions[15] = 0.0f;
                #pragma unroll
                for (int route = 0; route < TOP_K; route++) {
                    int target_rank = target_ranks[route];
                    int send_index = send_indices[route];
                    if (target_rank >= 0 && send_index >= 0) {
                        unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                        unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                        if (payload_dtype_code == 0 || payload_dtype_code == 1) {
                            unsigned int pair_word[1];
                            {
                                pair_word[0] = *reinterpret_cast<const unsigned int*>(workspace_u32 + (byte_index / 4ull));
                            }
                            if (payload_dtype_code == 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    (&contributions[2 * route + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) << 16);
                                    (&contributions[2 * route + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) & 0xffff0000u);
                                }
                            } else {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 h_lo, h_hi;\n\t"
                                        ".reg .b32 f_lo, f_hi;\n\t"
                                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&contributions[2 * route + _pair * 2]))
                                        : "r"(pair_word[_pair]));
                                }
                            }
                        }
                        if (payload_dtype_code == 2) {
                            {
                                uint16_t _e4m3x2_0 = *reinterpret_cast<const uint16_t*>(workspace_fp8 + byte_index);
                                uint32_t _f16x2_0;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                                uint16_t _h0_0 = (uint16_t)((_f16x2_0 >> 0) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 0]) : "h"(_h0_0));
                                uint16_t _h1_0 = (uint16_t)((_f16x2_0 >> 16) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 1]) : "h"(_h1_0));
                            }
                        }
                        if (payload_dtype_code == 3) {
                            {
                                float2 _v2_1 = *reinterpret_cast<const float2*>(workspace_f32 + byte_index / 4ull);
                                contributions[2 * route] = _v2_1.x;
                                contributions[2 * route + 1] = _v2_1.y;
                            }
                        }
                        if (use_low_precision) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_2) : "f"(contributions[2 * route + 1]), "f"(contributions[2 * route]));
                            float2 rounded;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(rounded.x), "=f"(rounded.y) : "h"(_cvt_satfinite_e4m3x2_f32_2));
                            contributions[2 * route] = rounded.x;
                            contributions[2 * route + 1] = rounded.y;
                        }
                    }
                }
                contributions[0] = contributions[0] + contributions[2];
                contributions[4] = contributions[4] + contributions[6];
                contributions[8] = contributions[8] + contributions[10];
                contributions[12] = contributions[12] + contributions[14];
                contributions[0] = contributions[0] + contributions[4];
                contributions[8] = contributions[8] + contributions[12];
                contributions[0] = contributions[0] + contributions[8];
                float result_0 = contributions[0];
                if (payload_dtype_code == 1) {
                    carrier_values[0] = (float)(__half)result_0;
                } else {
                    carrier_values[0] = (float)(__nv_bfloat16)result_0;
                }
                contributions[1] = contributions[1] + contributions[3];
                contributions[5] = contributions[5] + contributions[7];
                contributions[9] = contributions[9] + contributions[11];
                contributions[13] = contributions[13] + contributions[15];
                contributions[1] = contributions[1] + contributions[5];
                contributions[9] = contributions[9] + contributions[13];
                contributions[1] = contributions[1] + contributions[9];
                float result_1 = contributions[1];
                if (payload_dtype_code == 1) {
                    carrier_values[1] = (float)(__half)result_1;
                } else {
                    carrier_values[1] = (float)(__nv_bfloat16)result_1;
                }
            }
            int block_column = column / block_size;
            float block_max;
            asm("max.f32 %0, %1, %2;" : "=f"(block_max) : "f"(fabsf(carrier_values[0])), "f"(fabsf(carrier_values[1])));
            if (quant_mode == 3) {
                block_max = max_noftz(block_max, 0.0f);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_5);
            } else {
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
                block_max = max_noftz(block_max, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_8);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_9);
            }
            int scale_byte = 0;
            float actual_scale = 0.0f;
            float fp8_scale = 0.0f;
            if (quant_mode == 3) {
                float reciprocal_six = approx_rcp(6.0f);
                float sf_value = output_scalar_scale * (block_max * reciprocal_six);
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
                uint32_t _f16x2_decode_13;
                float _fp8_e4m3_decode_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
                fp8_scale = _fp8_e4m3_decode_10;
                if (block_max != 0.0f) {
                    float reciprocal_scalar = approx_rcp(output_scalar_scale);
                    float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
                    actual_scale = _rcp_14;
                }
            } else {
                float denominator = 6.0f;
                if (quant_mode == 1) {
                    denominator = 448.0f;
                }
                float _rcp_15 = approx_rcp(denominator);
                float raw_scale = block_max * _rcp_15;
                int raw_bits = 0;
                raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
                int exponent = raw_bits >> 23 & 255;
                int mantissa = raw_bits & 8388607;
                int round_up = 0;
                if (mantissa != 0) {
                    if (exponent != 0) {
                        round_up = 1;
                    } else if (mantissa > 4194304) {
                        round_up = 1;
                    }
                }
                if (raw_scale > 0.0f) {
                    scale_byte = exponent + round_up;
                    if (scale_byte > 254) {
                        scale_byte = 254;
                    }
                }
                if (quant_mode == 1) {
                    int scale_bits = scale_byte << 23;
                    if (scale_byte == 0) {
                        scale_bits = 4194304;
                    }
                    float decoded_scale = 0.0f;
                    decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
                    float _rcp_16 = approx_rcp(decoded_scale);
                    actual_scale = _rcp_16;
                } else if (block_max != 0.0f) {
                    if (scale_byte == 0) {
                        actual_scale = 1.0f;
                    } else {
                        float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                        actual_scale = _exp2_17;
                    }
                }
            }
            int scale_index = token * blocks_per_row + block_column;
            if (scale_layout == 0) {
                scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
            }
            if (scale_layout == 1) {
                int tiles = padded_scale_cols / 4;
                scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
            }
            int block_lane = lane & 15;
            if (quant_mode == 3) {
                block_lane = lane & 7;
            }
            if (active != 0) {
                if (block_lane == 0) {
                    if (quant_mode == 3) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                        *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
                    } else {
                        scales_u8[scale_index] = scale_byte;
                    }
                }
                float normalized_low = carrier_values[0] * actual_scale;
                float normalized_high = carrier_values[1] * actual_scale;
                if (quant_mode == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(normalized_high), "f"(normalized_low));
                    *reinterpret_cast<unsigned short*>(quantized_fp8 + (token * elements_per_token + column)) = (unsigned short)(_cvt_satfinite_e4m3x2_f32_19);
                } else {
                    uint32_t _fp4_e2m1_pack_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_e2m1_pack_pair_20) : "f"(normalized_low), "f"(normalized_high));
                    quantized_packed[token * (elements_per_token / 2) + column / 2] = _fp4_e2m1_pack_pair_20;
                }
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 10

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_10(uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    uint8_t* workspace_fp8 = reinterpret_cast<uint8_t*>(workspace);
    float* workspace_f32 = reinterpret_cast<float*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[10];
        int send_indices[10];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column_base = 0; column_base < elements_per_token; column_base += 512) {
            int column = column_base + 2 * tid;
            int active = 0;
            float carrier_values[2];
            carrier_values[0] = 0.0f;
            carrier_values[1] = 0.0f;
            if (column < elements_per_token) {
                active = 1;
                float contributions[20];
                contributions[0] = 0.0f;
                contributions[1] = 0.0f;
                contributions[2] = 0.0f;
                contributions[3] = 0.0f;
                contributions[4] = 0.0f;
                contributions[5] = 0.0f;
                contributions[6] = 0.0f;
                contributions[7] = 0.0f;
                contributions[8] = 0.0f;
                contributions[9] = 0.0f;
                contributions[10] = 0.0f;
                contributions[11] = 0.0f;
                contributions[12] = 0.0f;
                contributions[13] = 0.0f;
                contributions[14] = 0.0f;
                contributions[15] = 0.0f;
                contributions[16] = 0.0f;
                contributions[17] = 0.0f;
                contributions[18] = 0.0f;
                contributions[19] = 0.0f;
                #pragma unroll
                for (int route = 0; route < TOP_K; route++) {
                    int target_rank = target_ranks[route];
                    int send_index = send_indices[route];
                    if (target_rank >= 0 && send_index >= 0) {
                        unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                        unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                        if (payload_dtype_code == 0 || payload_dtype_code == 1) {
                            unsigned int pair_word[1];
                            {
                                pair_word[0] = *reinterpret_cast<const unsigned int*>(workspace_u32 + (byte_index / 4ull));
                            }
                            if (payload_dtype_code == 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    (&contributions[2 * route + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) << 16);
                                    (&contributions[2 * route + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) & 0xffff0000u);
                                }
                            } else {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 h_lo, h_hi;\n\t"
                                        ".reg .b32 f_lo, f_hi;\n\t"
                                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&contributions[2 * route + _pair * 2]))
                                        : "r"(pair_word[_pair]));
                                }
                            }
                        }
                        if (payload_dtype_code == 2) {
                            {
                                uint16_t _e4m3x2_0 = *reinterpret_cast<const uint16_t*>(workspace_fp8 + byte_index);
                                uint32_t _f16x2_0;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                                uint16_t _h0_0 = (uint16_t)((_f16x2_0 >> 0) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 0]) : "h"(_h0_0));
                                uint16_t _h1_0 = (uint16_t)((_f16x2_0 >> 16) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 1]) : "h"(_h1_0));
                            }
                        }
                        if (payload_dtype_code == 3) {
                            {
                                float2 _v2_1 = *reinterpret_cast<const float2*>(workspace_f32 + byte_index / 4ull);
                                contributions[2 * route] = _v2_1.x;
                                contributions[2 * route + 1] = _v2_1.y;
                            }
                        }
                        if (use_low_precision) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_2) : "f"(contributions[2 * route + 1]), "f"(contributions[2 * route]));
                            float2 rounded;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(rounded.x), "=f"(rounded.y) : "h"(_cvt_satfinite_e4m3x2_f32_2));
                            contributions[2 * route] = rounded.x;
                            contributions[2 * route + 1] = rounded.y;
                        }
                    }
                }
                contributions[0] = contributions[0] + contributions[2];
                contributions[4] = contributions[4] + contributions[6];
                contributions[8] = contributions[8] + contributions[10];
                contributions[12] = contributions[12] + contributions[14];
                contributions[16] = contributions[16] + contributions[18];
                contributions[0] = contributions[0] + contributions[4];
                contributions[8] = contributions[8] + contributions[12];
                contributions[0] = contributions[0] + contributions[8];
                contributions[0] = contributions[0] + contributions[16];
                float result_0 = contributions[0];
                if (payload_dtype_code == 1) {
                    carrier_values[0] = (float)(__half)result_0;
                } else {
                    carrier_values[0] = (float)(__nv_bfloat16)result_0;
                }
                contributions[1] = contributions[1] + contributions[3];
                contributions[5] = contributions[5] + contributions[7];
                contributions[9] = contributions[9] + contributions[11];
                contributions[13] = contributions[13] + contributions[15];
                contributions[17] = contributions[17] + contributions[19];
                contributions[1] = contributions[1] + contributions[5];
                contributions[9] = contributions[9] + contributions[13];
                contributions[1] = contributions[1] + contributions[9];
                contributions[1] = contributions[1] + contributions[17];
                float result_1 = contributions[1];
                if (payload_dtype_code == 1) {
                    carrier_values[1] = (float)(__half)result_1;
                } else {
                    carrier_values[1] = (float)(__nv_bfloat16)result_1;
                }
            }
            int block_column = column / block_size;
            float block_max;
            asm("max.f32 %0, %1, %2;" : "=f"(block_max) : "f"(fabsf(carrier_values[0])), "f"(fabsf(carrier_values[1])));
            if (quant_mode == 3) {
                block_max = max_noftz(block_max, 0.0f);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_5);
            } else {
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
                block_max = max_noftz(block_max, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_8);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_9);
            }
            int scale_byte = 0;
            float actual_scale = 0.0f;
            float fp8_scale = 0.0f;
            if (quant_mode == 3) {
                float reciprocal_six = approx_rcp(6.0f);
                float sf_value = output_scalar_scale * (block_max * reciprocal_six);
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
                uint32_t _f16x2_decode_13;
                float _fp8_e4m3_decode_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
                fp8_scale = _fp8_e4m3_decode_10;
                if (block_max != 0.0f) {
                    float reciprocal_scalar = approx_rcp(output_scalar_scale);
                    float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
                    actual_scale = _rcp_14;
                }
            } else {
                float denominator = 6.0f;
                if (quant_mode == 1) {
                    denominator = 448.0f;
                }
                float _rcp_15 = approx_rcp(denominator);
                float raw_scale = block_max * _rcp_15;
                int raw_bits = 0;
                raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
                int exponent = raw_bits >> 23 & 255;
                int mantissa = raw_bits & 8388607;
                int round_up = 0;
                if (mantissa != 0) {
                    if (exponent != 0) {
                        round_up = 1;
                    } else if (mantissa > 4194304) {
                        round_up = 1;
                    }
                }
                if (raw_scale > 0.0f) {
                    scale_byte = exponent + round_up;
                    if (scale_byte > 254) {
                        scale_byte = 254;
                    }
                }
                if (quant_mode == 1) {
                    int scale_bits = scale_byte << 23;
                    if (scale_byte == 0) {
                        scale_bits = 4194304;
                    }
                    float decoded_scale = 0.0f;
                    decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
                    float _rcp_16 = approx_rcp(decoded_scale);
                    actual_scale = _rcp_16;
                } else if (block_max != 0.0f) {
                    if (scale_byte == 0) {
                        actual_scale = 1.0f;
                    } else {
                        float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                        actual_scale = _exp2_17;
                    }
                }
            }
            int scale_index = token * blocks_per_row + block_column;
            if (scale_layout == 0) {
                scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
            }
            if (scale_layout == 1) {
                int tiles = padded_scale_cols / 4;
                scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
            }
            int block_lane = lane & 15;
            if (quant_mode == 3) {
                block_lane = lane & 7;
            }
            if (active != 0) {
                if (block_lane == 0) {
                    if (quant_mode == 3) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                        *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
                    } else {
                        scales_u8[scale_index] = scale_byte;
                    }
                }
                float normalized_low = carrier_values[0] * actual_scale;
                float normalized_high = carrier_values[1] * actual_scale;
                if (quant_mode == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(normalized_high), "f"(normalized_low));
                    *reinterpret_cast<unsigned short*>(quantized_fp8 + (token * elements_per_token + column)) = (unsigned short)(_cvt_satfinite_e4m3x2_f32_19);
                } else {
                    uint32_t _fp4_e2m1_pack_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_e2m1_pack_pair_20) : "f"(normalized_low), "f"(normalized_high));
                    quantized_packed[token * (elements_per_token / 2) + column / 2] = _fp4_e2m1_pack_pair_20;
                }
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 12

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_12(uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    uint8_t* workspace_fp8 = reinterpret_cast<uint8_t*>(workspace);
    float* workspace_f32 = reinterpret_cast<float*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[12];
        int send_indices[12];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column_base = 0; column_base < elements_per_token; column_base += 512) {
            int column = column_base + 2 * tid;
            int active = 0;
            float carrier_values[2];
            carrier_values[0] = 0.0f;
            carrier_values[1] = 0.0f;
            if (column < elements_per_token) {
                active = 1;
                float contributions[24];
                contributions[0] = 0.0f;
                contributions[1] = 0.0f;
                contributions[2] = 0.0f;
                contributions[3] = 0.0f;
                contributions[4] = 0.0f;
                contributions[5] = 0.0f;
                contributions[6] = 0.0f;
                contributions[7] = 0.0f;
                contributions[8] = 0.0f;
                contributions[9] = 0.0f;
                contributions[10] = 0.0f;
                contributions[11] = 0.0f;
                contributions[12] = 0.0f;
                contributions[13] = 0.0f;
                contributions[14] = 0.0f;
                contributions[15] = 0.0f;
                contributions[16] = 0.0f;
                contributions[17] = 0.0f;
                contributions[18] = 0.0f;
                contributions[19] = 0.0f;
                contributions[20] = 0.0f;
                contributions[21] = 0.0f;
                contributions[22] = 0.0f;
                contributions[23] = 0.0f;
                #pragma unroll
                for (int route = 0; route < TOP_K; route++) {
                    int target_rank = target_ranks[route];
                    int send_index = send_indices[route];
                    if (target_rank >= 0 && send_index >= 0) {
                        unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                        unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                        if (payload_dtype_code == 0 || payload_dtype_code == 1) {
                            unsigned int pair_word[1];
                            {
                                pair_word[0] = *reinterpret_cast<const unsigned int*>(workspace_u32 + (byte_index / 4ull));
                            }
                            if (payload_dtype_code == 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    (&contributions[2 * route + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) << 16);
                                    (&contributions[2 * route + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) & 0xffff0000u);
                                }
                            } else {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 h_lo, h_hi;\n\t"
                                        ".reg .b32 f_lo, f_hi;\n\t"
                                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&contributions[2 * route + _pair * 2]))
                                        : "r"(pair_word[_pair]));
                                }
                            }
                        }
                        if (payload_dtype_code == 2) {
                            {
                                uint16_t _e4m3x2_0 = *reinterpret_cast<const uint16_t*>(workspace_fp8 + byte_index);
                                uint32_t _f16x2_0;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                                uint16_t _h0_0 = (uint16_t)((_f16x2_0 >> 0) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 0]) : "h"(_h0_0));
                                uint16_t _h1_0 = (uint16_t)((_f16x2_0 >> 16) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 1]) : "h"(_h1_0));
                            }
                        }
                        if (payload_dtype_code == 3) {
                            {
                                float2 _v2_1 = *reinterpret_cast<const float2*>(workspace_f32 + byte_index / 4ull);
                                contributions[2 * route] = _v2_1.x;
                                contributions[2 * route + 1] = _v2_1.y;
                            }
                        }
                        if (use_low_precision) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_2) : "f"(contributions[2 * route + 1]), "f"(contributions[2 * route]));
                            float2 rounded;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(rounded.x), "=f"(rounded.y) : "h"(_cvt_satfinite_e4m3x2_f32_2));
                            contributions[2 * route] = rounded.x;
                            contributions[2 * route + 1] = rounded.y;
                        }
                    }
                }
                #pragma unroll
                for (int route_reduce_0 = 1; route_reduce_0 < TOP_K; route_reduce_0++) {
                    contributions[0] = contributions[0] + contributions[2 * route_reduce_0];
                }
                float result_0 = contributions[0];
                if (payload_dtype_code == 1) {
                    carrier_values[0] = (float)(__half)result_0;
                } else {
                    carrier_values[0] = (float)(__nv_bfloat16)result_0;
                }
                #pragma unroll
                for (int route_reduce_1 = 1; route_reduce_1 < TOP_K; route_reduce_1++) {
                    contributions[1] = contributions[1] + contributions[2 * route_reduce_1 + 1];
                }
                float result_1 = contributions[1];
                if (payload_dtype_code == 1) {
                    carrier_values[1] = (float)(__half)result_1;
                } else {
                    carrier_values[1] = (float)(__nv_bfloat16)result_1;
                }
            }
            int block_column = column / block_size;
            float block_max;
            asm("max.f32 %0, %1, %2;" : "=f"(block_max) : "f"(fabsf(carrier_values[0])), "f"(fabsf(carrier_values[1])));
            if (quant_mode == 3) {
                block_max = max_noftz(block_max, 0.0f);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_5);
            } else {
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
                block_max = max_noftz(block_max, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_8);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_9);
            }
            int scale_byte = 0;
            float actual_scale = 0.0f;
            float fp8_scale = 0.0f;
            if (quant_mode == 3) {
                float reciprocal_six = approx_rcp(6.0f);
                float sf_value = output_scalar_scale * (block_max * reciprocal_six);
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
                uint32_t _f16x2_decode_13;
                float _fp8_e4m3_decode_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
                fp8_scale = _fp8_e4m3_decode_10;
                if (block_max != 0.0f) {
                    float reciprocal_scalar = approx_rcp(output_scalar_scale);
                    float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
                    actual_scale = _rcp_14;
                }
            } else {
                float denominator = 6.0f;
                if (quant_mode == 1) {
                    denominator = 448.0f;
                }
                float _rcp_15 = approx_rcp(denominator);
                float raw_scale = block_max * _rcp_15;
                int raw_bits = 0;
                raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
                int exponent = raw_bits >> 23 & 255;
                int mantissa = raw_bits & 8388607;
                int round_up = 0;
                if (mantissa != 0) {
                    if (exponent != 0) {
                        round_up = 1;
                    } else if (mantissa > 4194304) {
                        round_up = 1;
                    }
                }
                if (raw_scale > 0.0f) {
                    scale_byte = exponent + round_up;
                    if (scale_byte > 254) {
                        scale_byte = 254;
                    }
                }
                if (quant_mode == 1) {
                    int scale_bits = scale_byte << 23;
                    if (scale_byte == 0) {
                        scale_bits = 4194304;
                    }
                    float decoded_scale = 0.0f;
                    decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
                    float _rcp_16 = approx_rcp(decoded_scale);
                    actual_scale = _rcp_16;
                } else if (block_max != 0.0f) {
                    if (scale_byte == 0) {
                        actual_scale = 1.0f;
                    } else {
                        float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                        actual_scale = _exp2_17;
                    }
                }
            }
            int scale_index = token * blocks_per_row + block_column;
            if (scale_layout == 0) {
                scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
            }
            if (scale_layout == 1) {
                int tiles = padded_scale_cols / 4;
                scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
            }
            int block_lane = lane & 15;
            if (quant_mode == 3) {
                block_lane = lane & 7;
            }
            if (active != 0) {
                if (block_lane == 0) {
                    if (quant_mode == 3) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                        *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
                    } else {
                        scales_u8[scale_index] = scale_byte;
                    }
                }
                float normalized_low = carrier_values[0] * actual_scale;
                float normalized_high = carrier_values[1] * actual_scale;
                if (quant_mode == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(normalized_high), "f"(normalized_low));
                    *reinterpret_cast<unsigned short*>(quantized_fp8 + (token * elements_per_token + column)) = (unsigned short)(_cvt_satfinite_e4m3x2_f32_19);
                } else {
                    uint32_t _fp4_e2m1_pack_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_e2m1_pack_pair_20) : "f"(normalized_low), "f"(normalized_high));
                    quantized_packed[token * (elements_per_token / 2) + column / 2] = _fp4_e2m1_pack_pair_20;
                }
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 14

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_14(uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    uint8_t* workspace_fp8 = reinterpret_cast<uint8_t*>(workspace);
    float* workspace_f32 = reinterpret_cast<float*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[14];
        int send_indices[14];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column_base = 0; column_base < elements_per_token; column_base += 512) {
            int column = column_base + 2 * tid;
            int active = 0;
            float carrier_values[2];
            carrier_values[0] = 0.0f;
            carrier_values[1] = 0.0f;
            if (column < elements_per_token) {
                active = 1;
                float contributions[28];
                contributions[0] = 0.0f;
                contributions[1] = 0.0f;
                contributions[2] = 0.0f;
                contributions[3] = 0.0f;
                contributions[4] = 0.0f;
                contributions[5] = 0.0f;
                contributions[6] = 0.0f;
                contributions[7] = 0.0f;
                contributions[8] = 0.0f;
                contributions[9] = 0.0f;
                contributions[10] = 0.0f;
                contributions[11] = 0.0f;
                contributions[12] = 0.0f;
                contributions[13] = 0.0f;
                contributions[14] = 0.0f;
                contributions[15] = 0.0f;
                contributions[16] = 0.0f;
                contributions[17] = 0.0f;
                contributions[18] = 0.0f;
                contributions[19] = 0.0f;
                contributions[20] = 0.0f;
                contributions[21] = 0.0f;
                contributions[22] = 0.0f;
                contributions[23] = 0.0f;
                contributions[24] = 0.0f;
                contributions[25] = 0.0f;
                contributions[26] = 0.0f;
                contributions[27] = 0.0f;
                #pragma unroll
                for (int route = 0; route < TOP_K; route++) {
                    int target_rank = target_ranks[route];
                    int send_index = send_indices[route];
                    if (target_rank >= 0 && send_index >= 0) {
                        unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                        unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                        if (payload_dtype_code == 0 || payload_dtype_code == 1) {
                            unsigned int pair_word[1];
                            {
                                pair_word[0] = *reinterpret_cast<const unsigned int*>(workspace_u32 + (byte_index / 4ull));
                            }
                            if (payload_dtype_code == 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    (&contributions[2 * route + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) << 16);
                                    (&contributions[2 * route + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) & 0xffff0000u);
                                }
                            } else {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 h_lo, h_hi;\n\t"
                                        ".reg .b32 f_lo, f_hi;\n\t"
                                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&contributions[2 * route + _pair * 2]))
                                        : "r"(pair_word[_pair]));
                                }
                            }
                        }
                        if (payload_dtype_code == 2) {
                            {
                                uint16_t _e4m3x2_0 = *reinterpret_cast<const uint16_t*>(workspace_fp8 + byte_index);
                                uint32_t _f16x2_0;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                                uint16_t _h0_0 = (uint16_t)((_f16x2_0 >> 0) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 0]) : "h"(_h0_0));
                                uint16_t _h1_0 = (uint16_t)((_f16x2_0 >> 16) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 1]) : "h"(_h1_0));
                            }
                        }
                        if (payload_dtype_code == 3) {
                            {
                                float2 _v2_1 = *reinterpret_cast<const float2*>(workspace_f32 + byte_index / 4ull);
                                contributions[2 * route] = _v2_1.x;
                                contributions[2 * route + 1] = _v2_1.y;
                            }
                        }
                        if (use_low_precision) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_2) : "f"(contributions[2 * route + 1]), "f"(contributions[2 * route]));
                            float2 rounded;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(rounded.x), "=f"(rounded.y) : "h"(_cvt_satfinite_e4m3x2_f32_2));
                            contributions[2 * route] = rounded.x;
                            contributions[2 * route + 1] = rounded.y;
                        }
                    }
                }
                #pragma unroll
                for (int route_reduce_0 = 1; route_reduce_0 < TOP_K; route_reduce_0++) {
                    contributions[0] = contributions[0] + contributions[2 * route_reduce_0];
                }
                float result_0 = contributions[0];
                if (payload_dtype_code == 1) {
                    carrier_values[0] = (float)(__half)result_0;
                } else {
                    carrier_values[0] = (float)(__nv_bfloat16)result_0;
                }
                #pragma unroll
                for (int route_reduce_1 = 1; route_reduce_1 < TOP_K; route_reduce_1++) {
                    contributions[1] = contributions[1] + contributions[2 * route_reduce_1 + 1];
                }
                float result_1 = contributions[1];
                if (payload_dtype_code == 1) {
                    carrier_values[1] = (float)(__half)result_1;
                } else {
                    carrier_values[1] = (float)(__nv_bfloat16)result_1;
                }
            }
            int block_column = column / block_size;
            float block_max;
            asm("max.f32 %0, %1, %2;" : "=f"(block_max) : "f"(fabsf(carrier_values[0])), "f"(fabsf(carrier_values[1])));
            if (quant_mode == 3) {
                block_max = max_noftz(block_max, 0.0f);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_5);
            } else {
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
                block_max = max_noftz(block_max, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_8);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_9);
            }
            int scale_byte = 0;
            float actual_scale = 0.0f;
            float fp8_scale = 0.0f;
            if (quant_mode == 3) {
                float reciprocal_six = approx_rcp(6.0f);
                float sf_value = output_scalar_scale * (block_max * reciprocal_six);
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
                uint32_t _f16x2_decode_13;
                float _fp8_e4m3_decode_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
                fp8_scale = _fp8_e4m3_decode_10;
                if (block_max != 0.0f) {
                    float reciprocal_scalar = approx_rcp(output_scalar_scale);
                    float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
                    actual_scale = _rcp_14;
                }
            } else {
                float denominator = 6.0f;
                if (quant_mode == 1) {
                    denominator = 448.0f;
                }
                float _rcp_15 = approx_rcp(denominator);
                float raw_scale = block_max * _rcp_15;
                int raw_bits = 0;
                raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
                int exponent = raw_bits >> 23 & 255;
                int mantissa = raw_bits & 8388607;
                int round_up = 0;
                if (mantissa != 0) {
                    if (exponent != 0) {
                        round_up = 1;
                    } else if (mantissa > 4194304) {
                        round_up = 1;
                    }
                }
                if (raw_scale > 0.0f) {
                    scale_byte = exponent + round_up;
                    if (scale_byte > 254) {
                        scale_byte = 254;
                    }
                }
                if (quant_mode == 1) {
                    int scale_bits = scale_byte << 23;
                    if (scale_byte == 0) {
                        scale_bits = 4194304;
                    }
                    float decoded_scale = 0.0f;
                    decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
                    float _rcp_16 = approx_rcp(decoded_scale);
                    actual_scale = _rcp_16;
                } else if (block_max != 0.0f) {
                    if (scale_byte == 0) {
                        actual_scale = 1.0f;
                    } else {
                        float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                        actual_scale = _exp2_17;
                    }
                }
            }
            int scale_index = token * blocks_per_row + block_column;
            if (scale_layout == 0) {
                scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
            }
            if (scale_layout == 1) {
                int tiles = padded_scale_cols / 4;
                scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
            }
            int block_lane = lane & 15;
            if (quant_mode == 3) {
                block_lane = lane & 7;
            }
            if (active != 0) {
                if (block_lane == 0) {
                    if (quant_mode == 3) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                        *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
                    } else {
                        scales_u8[scale_index] = scale_byte;
                    }
                }
                float normalized_low = carrier_values[0] * actual_scale;
                float normalized_high = carrier_values[1] * actual_scale;
                if (quant_mode == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(normalized_high), "f"(normalized_low));
                    *reinterpret_cast<unsigned short*>(quantized_fp8 + (token * elements_per_token + column)) = (unsigned short)(_cvt_satfinite_e4m3x2_f32_19);
                } else {
                    uint32_t _fp4_e2m1_pack_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_e2m1_pack_pair_20) : "f"(normalized_low), "f"(normalized_high));
                    quantized_packed[token * (elements_per_token / 2) + column / 2] = _fp4_e2m1_pack_pair_20;
                }
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 16

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_16(uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    uint8_t* workspace_fp8 = reinterpret_cast<uint8_t*>(workspace);
    float* workspace_f32 = reinterpret_cast<float*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[16];
        int send_indices[16];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column_base = 0; column_base < elements_per_token; column_base += 512) {
            int column = column_base + 2 * tid;
            int active = 0;
            float carrier_values[2];
            carrier_values[0] = 0.0f;
            carrier_values[1] = 0.0f;
            if (column < elements_per_token) {
                active = 1;
                float contributions[32];
                contributions[0] = 0.0f;
                contributions[1] = 0.0f;
                contributions[2] = 0.0f;
                contributions[3] = 0.0f;
                contributions[4] = 0.0f;
                contributions[5] = 0.0f;
                contributions[6] = 0.0f;
                contributions[7] = 0.0f;
                contributions[8] = 0.0f;
                contributions[9] = 0.0f;
                contributions[10] = 0.0f;
                contributions[11] = 0.0f;
                contributions[12] = 0.0f;
                contributions[13] = 0.0f;
                contributions[14] = 0.0f;
                contributions[15] = 0.0f;
                contributions[16] = 0.0f;
                contributions[17] = 0.0f;
                contributions[18] = 0.0f;
                contributions[19] = 0.0f;
                contributions[20] = 0.0f;
                contributions[21] = 0.0f;
                contributions[22] = 0.0f;
                contributions[23] = 0.0f;
                contributions[24] = 0.0f;
                contributions[25] = 0.0f;
                contributions[26] = 0.0f;
                contributions[27] = 0.0f;
                contributions[28] = 0.0f;
                contributions[29] = 0.0f;
                contributions[30] = 0.0f;
                contributions[31] = 0.0f;
                #pragma unroll
                for (int route = 0; route < TOP_K; route++) {
                    int target_rank = target_ranks[route];
                    int send_index = send_indices[route];
                    if (target_rank >= 0 && send_index >= 0) {
                        unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                        unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                        if (payload_dtype_code == 0 || payload_dtype_code == 1) {
                            unsigned int pair_word[1];
                            {
                                pair_word[0] = *reinterpret_cast<const unsigned int*>(workspace_u32 + (byte_index / 4ull));
                            }
                            if (payload_dtype_code == 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    (&contributions[2 * route + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) << 16);
                                    (&contributions[2 * route + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) & 0xffff0000u);
                                }
                            } else {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 h_lo, h_hi;\n\t"
                                        ".reg .b32 f_lo, f_hi;\n\t"
                                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&contributions[2 * route + _pair * 2]))
                                        : "r"(pair_word[_pair]));
                                }
                            }
                        }
                        if (payload_dtype_code == 2) {
                            {
                                uint16_t _e4m3x2_0 = *reinterpret_cast<const uint16_t*>(workspace_fp8 + byte_index);
                                uint32_t _f16x2_0;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                                uint16_t _h0_0 = (uint16_t)((_f16x2_0 >> 0) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 0]) : "h"(_h0_0));
                                uint16_t _h1_0 = (uint16_t)((_f16x2_0 >> 16) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 1]) : "h"(_h1_0));
                            }
                        }
                        if (payload_dtype_code == 3) {
                            {
                                float2 _v2_1 = *reinterpret_cast<const float2*>(workspace_f32 + byte_index / 4ull);
                                contributions[2 * route] = _v2_1.x;
                                contributions[2 * route + 1] = _v2_1.y;
                            }
                        }
                        if (use_low_precision) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_2) : "f"(contributions[2 * route + 1]), "f"(contributions[2 * route]));
                            float2 rounded;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(rounded.x), "=f"(rounded.y) : "h"(_cvt_satfinite_e4m3x2_f32_2));
                            contributions[2 * route] = rounded.x;
                            contributions[2 * route + 1] = rounded.y;
                        }
                    }
                }
                contributions[0] = contributions[0] + contributions[2];
                contributions[4] = contributions[4] + contributions[6];
                contributions[8] = contributions[8] + contributions[10];
                contributions[12] = contributions[12] + contributions[14];
                contributions[16] = contributions[16] + contributions[18];
                contributions[20] = contributions[20] + contributions[22];
                contributions[24] = contributions[24] + contributions[26];
                contributions[28] = contributions[28] + contributions[30];
                contributions[0] = contributions[0] + contributions[4];
                contributions[8] = contributions[8] + contributions[12];
                contributions[16] = contributions[16] + contributions[20];
                contributions[24] = contributions[24] + contributions[28];
                contributions[0] = contributions[0] + contributions[8];
                contributions[16] = contributions[16] + contributions[24];
                contributions[0] = contributions[0] + contributions[16];
                float result_0 = contributions[0];
                if (payload_dtype_code == 1) {
                    carrier_values[0] = (float)(__half)result_0;
                } else {
                    carrier_values[0] = (float)(__nv_bfloat16)result_0;
                }
                contributions[1] = contributions[1] + contributions[3];
                contributions[5] = contributions[5] + contributions[7];
                contributions[9] = contributions[9] + contributions[11];
                contributions[13] = contributions[13] + contributions[15];
                contributions[17] = contributions[17] + contributions[19];
                contributions[21] = contributions[21] + contributions[23];
                contributions[25] = contributions[25] + contributions[27];
                contributions[29] = contributions[29] + contributions[31];
                contributions[1] = contributions[1] + contributions[5];
                contributions[9] = contributions[9] + contributions[13];
                contributions[17] = contributions[17] + contributions[21];
                contributions[25] = contributions[25] + contributions[29];
                contributions[1] = contributions[1] + contributions[9];
                contributions[17] = contributions[17] + contributions[25];
                contributions[1] = contributions[1] + contributions[17];
                float result_1 = contributions[1];
                if (payload_dtype_code == 1) {
                    carrier_values[1] = (float)(__half)result_1;
                } else {
                    carrier_values[1] = (float)(__nv_bfloat16)result_1;
                }
            }
            int block_column = column / block_size;
            float block_max;
            asm("max.f32 %0, %1, %2;" : "=f"(block_max) : "f"(fabsf(carrier_values[0])), "f"(fabsf(carrier_values[1])));
            if (quant_mode == 3) {
                block_max = max_noftz(block_max, 0.0f);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_5);
            } else {
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
                block_max = max_noftz(block_max, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_8);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_9);
            }
            int scale_byte = 0;
            float actual_scale = 0.0f;
            float fp8_scale = 0.0f;
            if (quant_mode == 3) {
                float reciprocal_six = approx_rcp(6.0f);
                float sf_value = output_scalar_scale * (block_max * reciprocal_six);
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
                uint32_t _f16x2_decode_13;
                float _fp8_e4m3_decode_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
                fp8_scale = _fp8_e4m3_decode_10;
                if (block_max != 0.0f) {
                    float reciprocal_scalar = approx_rcp(output_scalar_scale);
                    float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
                    actual_scale = _rcp_14;
                }
            } else {
                float denominator = 6.0f;
                if (quant_mode == 1) {
                    denominator = 448.0f;
                }
                float _rcp_15 = approx_rcp(denominator);
                float raw_scale = block_max * _rcp_15;
                int raw_bits = 0;
                raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
                int exponent = raw_bits >> 23 & 255;
                int mantissa = raw_bits & 8388607;
                int round_up = 0;
                if (mantissa != 0) {
                    if (exponent != 0) {
                        round_up = 1;
                    } else if (mantissa > 4194304) {
                        round_up = 1;
                    }
                }
                if (raw_scale > 0.0f) {
                    scale_byte = exponent + round_up;
                    if (scale_byte > 254) {
                        scale_byte = 254;
                    }
                }
                if (quant_mode == 1) {
                    int scale_bits = scale_byte << 23;
                    if (scale_byte == 0) {
                        scale_bits = 4194304;
                    }
                    float decoded_scale = 0.0f;
                    decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
                    float _rcp_16 = approx_rcp(decoded_scale);
                    actual_scale = _rcp_16;
                } else if (block_max != 0.0f) {
                    if (scale_byte == 0) {
                        actual_scale = 1.0f;
                    } else {
                        float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                        actual_scale = _exp2_17;
                    }
                }
            }
            int scale_index = token * blocks_per_row + block_column;
            if (scale_layout == 0) {
                scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
            }
            if (scale_layout == 1) {
                int tiles = padded_scale_cols / 4;
                scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
            }
            int block_lane = lane & 15;
            if (quant_mode == 3) {
                block_lane = lane & 7;
            }
            if (active != 0) {
                if (block_lane == 0) {
                    if (quant_mode == 3) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                        *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
                    } else {
                        scales_u8[scale_index] = scale_byte;
                    }
                }
                float normalized_low = carrier_values[0] * actual_scale;
                float normalized_high = carrier_values[1] * actual_scale;
                if (quant_mode == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(normalized_high), "f"(normalized_low));
                    *reinterpret_cast<unsigned short*>(quantized_fp8 + (token * elements_per_token + column)) = (unsigned short)(_cvt_satfinite_e4m3x2_f32_19);
                } else {
                    uint32_t _fp4_e2m1_pack_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_e2m1_pack_pair_20) : "f"(normalized_low), "f"(normalized_high));
                    quantized_packed[token * (elements_per_token / 2) + column / 2] = _fp4_e2m1_pack_pair_20;
                }
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 18

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_18(uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    uint8_t* workspace_fp8 = reinterpret_cast<uint8_t*>(workspace);
    float* workspace_f32 = reinterpret_cast<float*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[18];
        int send_indices[18];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column_base = 0; column_base < elements_per_token; column_base += 512) {
            int column = column_base + 2 * tid;
            int active = 0;
            float carrier_values[2];
            carrier_values[0] = 0.0f;
            carrier_values[1] = 0.0f;
            if (column < elements_per_token) {
                active = 1;
                float contributions[36];
                contributions[0] = 0.0f;
                contributions[1] = 0.0f;
                contributions[2] = 0.0f;
                contributions[3] = 0.0f;
                contributions[4] = 0.0f;
                contributions[5] = 0.0f;
                contributions[6] = 0.0f;
                contributions[7] = 0.0f;
                contributions[8] = 0.0f;
                contributions[9] = 0.0f;
                contributions[10] = 0.0f;
                contributions[11] = 0.0f;
                contributions[12] = 0.0f;
                contributions[13] = 0.0f;
                contributions[14] = 0.0f;
                contributions[15] = 0.0f;
                contributions[16] = 0.0f;
                contributions[17] = 0.0f;
                contributions[18] = 0.0f;
                contributions[19] = 0.0f;
                contributions[20] = 0.0f;
                contributions[21] = 0.0f;
                contributions[22] = 0.0f;
                contributions[23] = 0.0f;
                contributions[24] = 0.0f;
                contributions[25] = 0.0f;
                contributions[26] = 0.0f;
                contributions[27] = 0.0f;
                contributions[28] = 0.0f;
                contributions[29] = 0.0f;
                contributions[30] = 0.0f;
                contributions[31] = 0.0f;
                contributions[32] = 0.0f;
                contributions[33] = 0.0f;
                contributions[34] = 0.0f;
                contributions[35] = 0.0f;
                #pragma unroll
                for (int route = 0; route < TOP_K; route++) {
                    int target_rank = target_ranks[route];
                    int send_index = send_indices[route];
                    if (target_rank >= 0 && send_index >= 0) {
                        unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                        unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                        if (payload_dtype_code == 0 || payload_dtype_code == 1) {
                            unsigned int pair_word[1];
                            {
                                pair_word[0] = *reinterpret_cast<const unsigned int*>(workspace_u32 + (byte_index / 4ull));
                            }
                            if (payload_dtype_code == 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    (&contributions[2 * route + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) << 16);
                                    (&contributions[2 * route + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) & 0xffff0000u);
                                }
                            } else {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 h_lo, h_hi;\n\t"
                                        ".reg .b32 f_lo, f_hi;\n\t"
                                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&contributions[2 * route + _pair * 2]))
                                        : "r"(pair_word[_pair]));
                                }
                            }
                        }
                        if (payload_dtype_code == 2) {
                            {
                                uint16_t _e4m3x2_0 = *reinterpret_cast<const uint16_t*>(workspace_fp8 + byte_index);
                                uint32_t _f16x2_0;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                                uint16_t _h0_0 = (uint16_t)((_f16x2_0 >> 0) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 0]) : "h"(_h0_0));
                                uint16_t _h1_0 = (uint16_t)((_f16x2_0 >> 16) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 1]) : "h"(_h1_0));
                            }
                        }
                        if (payload_dtype_code == 3) {
                            {
                                float2 _v2_1 = *reinterpret_cast<const float2*>(workspace_f32 + byte_index / 4ull);
                                contributions[2 * route] = _v2_1.x;
                                contributions[2 * route + 1] = _v2_1.y;
                            }
                        }
                        if (use_low_precision) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_2) : "f"(contributions[2 * route + 1]), "f"(contributions[2 * route]));
                            float2 rounded;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(rounded.x), "=f"(rounded.y) : "h"(_cvt_satfinite_e4m3x2_f32_2));
                            contributions[2 * route] = rounded.x;
                            contributions[2 * route + 1] = rounded.y;
                        }
                    }
                }
                #pragma unroll
                for (int route_reduce_0 = 1; route_reduce_0 < TOP_K; route_reduce_0++) {
                    contributions[0] = contributions[0] + contributions[2 * route_reduce_0];
                }
                float result_0 = contributions[0];
                if (payload_dtype_code == 1) {
                    carrier_values[0] = (float)(__half)result_0;
                } else {
                    carrier_values[0] = (float)(__nv_bfloat16)result_0;
                }
                #pragma unroll
                for (int route_reduce_1 = 1; route_reduce_1 < TOP_K; route_reduce_1++) {
                    contributions[1] = contributions[1] + contributions[2 * route_reduce_1 + 1];
                }
                float result_1 = contributions[1];
                if (payload_dtype_code == 1) {
                    carrier_values[1] = (float)(__half)result_1;
                } else {
                    carrier_values[1] = (float)(__nv_bfloat16)result_1;
                }
            }
            int block_column = column / block_size;
            float block_max;
            asm("max.f32 %0, %1, %2;" : "=f"(block_max) : "f"(fabsf(carrier_values[0])), "f"(fabsf(carrier_values[1])));
            if (quant_mode == 3) {
                block_max = max_noftz(block_max, 0.0f);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_5);
            } else {
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
                block_max = max_noftz(block_max, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_8);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_9);
            }
            int scale_byte = 0;
            float actual_scale = 0.0f;
            float fp8_scale = 0.0f;
            if (quant_mode == 3) {
                float reciprocal_six = approx_rcp(6.0f);
                float sf_value = output_scalar_scale * (block_max * reciprocal_six);
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
                uint32_t _f16x2_decode_13;
                float _fp8_e4m3_decode_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
                fp8_scale = _fp8_e4m3_decode_10;
                if (block_max != 0.0f) {
                    float reciprocal_scalar = approx_rcp(output_scalar_scale);
                    float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
                    actual_scale = _rcp_14;
                }
            } else {
                float denominator = 6.0f;
                if (quant_mode == 1) {
                    denominator = 448.0f;
                }
                float _rcp_15 = approx_rcp(denominator);
                float raw_scale = block_max * _rcp_15;
                int raw_bits = 0;
                raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
                int exponent = raw_bits >> 23 & 255;
                int mantissa = raw_bits & 8388607;
                int round_up = 0;
                if (mantissa != 0) {
                    if (exponent != 0) {
                        round_up = 1;
                    } else if (mantissa > 4194304) {
                        round_up = 1;
                    }
                }
                if (raw_scale > 0.0f) {
                    scale_byte = exponent + round_up;
                    if (scale_byte > 254) {
                        scale_byte = 254;
                    }
                }
                if (quant_mode == 1) {
                    int scale_bits = scale_byte << 23;
                    if (scale_byte == 0) {
                        scale_bits = 4194304;
                    }
                    float decoded_scale = 0.0f;
                    decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
                    float _rcp_16 = approx_rcp(decoded_scale);
                    actual_scale = _rcp_16;
                } else if (block_max != 0.0f) {
                    if (scale_byte == 0) {
                        actual_scale = 1.0f;
                    } else {
                        float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                        actual_scale = _exp2_17;
                    }
                }
            }
            int scale_index = token * blocks_per_row + block_column;
            if (scale_layout == 0) {
                scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
            }
            if (scale_layout == 1) {
                int tiles = padded_scale_cols / 4;
                scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
            }
            int block_lane = lane & 15;
            if (quant_mode == 3) {
                block_lane = lane & 7;
            }
            if (active != 0) {
                if (block_lane == 0) {
                    if (quant_mode == 3) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                        *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
                    } else {
                        scales_u8[scale_index] = scale_byte;
                    }
                }
                float normalized_low = carrier_values[0] * actual_scale;
                float normalized_high = carrier_values[1] * actual_scale;
                if (quant_mode == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(normalized_high), "f"(normalized_low));
                    *reinterpret_cast<unsigned short*>(quantized_fp8 + (token * elements_per_token + column)) = (unsigned short)(_cvt_satfinite_e4m3x2_f32_19);
                } else {
                    uint32_t _fp4_e2m1_pack_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_e2m1_pack_pair_20) : "f"(normalized_low), "f"(normalized_high));
                    quantized_packed[token * (elements_per_token / 2) + column / 2] = _fp4_e2m1_pack_pair_20;
                }
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

#define TOP_K 22

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_quantized_top_k_22(uint8_t* __restrict__ workspace, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    uint8_t* workspace_fp8 = reinterpret_cast<uint8_t*>(workspace);
    float* workspace_f32 = reinterpret_cast<float*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int token = bid;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    if (token < local_num_tokens) {
        unsigned long long target_base = (local_workspace_base + topk_target_ranks_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        unsigned long long send_base = (local_workspace_base + topk_send_indices_offset) / 4ull + (unsigned long long)token * (unsigned long long)TOP_K;
        int target_ranks[22];
        int send_indices[22];
        #pragma unroll
        for (int route_load = 0; route_load < TOP_K; route_load++) {
            target_ranks[route_load] = workspace_i32[target_base + (unsigned long long)route_load];
            send_indices[route_load] = workspace_i32[send_base + (unsigned long long)route_load];
        }
        #pragma unroll 1
        for (int column_base = 0; column_base < elements_per_token; column_base += 512) {
            int column = column_base + 2 * tid;
            int active = 0;
            float carrier_values[2];
            carrier_values[0] = 0.0f;
            carrier_values[1] = 0.0f;
            if (column < elements_per_token) {
                active = 1;
                float contributions[44];
                contributions[0] = 0.0f;
                contributions[1] = 0.0f;
                contributions[2] = 0.0f;
                contributions[3] = 0.0f;
                contributions[4] = 0.0f;
                contributions[5] = 0.0f;
                contributions[6] = 0.0f;
                contributions[7] = 0.0f;
                contributions[8] = 0.0f;
                contributions[9] = 0.0f;
                contributions[10] = 0.0f;
                contributions[11] = 0.0f;
                contributions[12] = 0.0f;
                contributions[13] = 0.0f;
                contributions[14] = 0.0f;
                contributions[15] = 0.0f;
                contributions[16] = 0.0f;
                contributions[17] = 0.0f;
                contributions[18] = 0.0f;
                contributions[19] = 0.0f;
                contributions[20] = 0.0f;
                contributions[21] = 0.0f;
                contributions[22] = 0.0f;
                contributions[23] = 0.0f;
                contributions[24] = 0.0f;
                contributions[25] = 0.0f;
                contributions[26] = 0.0f;
                contributions[27] = 0.0f;
                contributions[28] = 0.0f;
                contributions[29] = 0.0f;
                contributions[30] = 0.0f;
                contributions[31] = 0.0f;
                contributions[32] = 0.0f;
                contributions[33] = 0.0f;
                contributions[34] = 0.0f;
                contributions[35] = 0.0f;
                contributions[36] = 0.0f;
                contributions[37] = 0.0f;
                contributions[38] = 0.0f;
                contributions[39] = 0.0f;
                contributions[40] = 0.0f;
                contributions[41] = 0.0f;
                contributions[42] = 0.0f;
                contributions[43] = 0.0f;
                #pragma unroll
                for (int route = 0; route < TOP_K; route++) {
                    int target_rank = target_ranks[route];
                    int send_index = send_indices[route];
                    if (target_rank >= 0 && send_index >= 0) {
                        unsigned long long payload_item = ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index) * (unsigned long long)elements_per_token + (unsigned long long)column;
                        unsigned long long byte_index = (unsigned long long)target_rank * workspace_stride_bytes + combine_payload_offset + payload_item * (unsigned long long)payload_element_bytes;
                        if (payload_dtype_code == 0 || payload_dtype_code == 1) {
                            unsigned int pair_word[1];
                            {
                                pair_word[0] = *reinterpret_cast<const unsigned int*>(workspace_u32 + (byte_index / 4ull));
                            }
                            if (payload_dtype_code == 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    (&contributions[2 * route + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) << 16);
                                    (&contributions[2 * route + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(pair_word[_pair]) & 0xffff0000u);
                                }
                            } else {
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 h_lo, h_hi;\n\t"
                                        ".reg .b32 f_lo, f_hi;\n\t"
                                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&contributions[2 * route + _pair * 2]))
                                        : "r"(pair_word[_pair]));
                                }
                            }
                        }
                        if (payload_dtype_code == 2) {
                            {
                                uint16_t _e4m3x2_0 = *reinterpret_cast<const uint16_t*>(workspace_fp8 + byte_index);
                                uint32_t _f16x2_0;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                                uint16_t _h0_0 = (uint16_t)((_f16x2_0 >> 0) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 0]) : "h"(_h0_0));
                                uint16_t _h1_0 = (uint16_t)((_f16x2_0 >> 16) & 0xFFFFu);
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(contributions[2 * route + 1]) : "h"(_h1_0));
                            }
                        }
                        if (payload_dtype_code == 3) {
                            {
                                float2 _v2_1 = *reinterpret_cast<const float2*>(workspace_f32 + byte_index / 4ull);
                                contributions[2 * route] = _v2_1.x;
                                contributions[2 * route + 1] = _v2_1.y;
                            }
                        }
                        if (use_low_precision) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_2) : "f"(contributions[2 * route + 1]), "f"(contributions[2 * route]));
                            float2 rounded;
                            asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
                                "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
                                "mov.b32 {lo, hi}, pair;\n"
                                "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
                                : "=f"(rounded.x), "=f"(rounded.y) : "h"(_cvt_satfinite_e4m3x2_f32_2));
                            contributions[2 * route] = rounded.x;
                            contributions[2 * route + 1] = rounded.y;
                        }
                    }
                }
                contributions[0] = contributions[0] + contributions[2];
                contributions[4] = contributions[4] + contributions[6];
                contributions[8] = contributions[8] + contributions[10];
                contributions[12] = contributions[12] + contributions[14];
                contributions[16] = contributions[16] + contributions[18];
                contributions[20] = contributions[20] + contributions[22];
                contributions[24] = contributions[24] + contributions[26];
                contributions[28] = contributions[28] + contributions[30];
                contributions[32] = contributions[32] + contributions[34];
                contributions[36] = contributions[36] + contributions[38];
                contributions[40] = contributions[40] + contributions[42];
                contributions[0] = contributions[0] + contributions[4];
                contributions[8] = contributions[8] + contributions[12];
                contributions[16] = contributions[16] + contributions[20];
                contributions[24] = contributions[24] + contributions[28];
                contributions[32] = contributions[32] + contributions[36];
                contributions[0] = contributions[0] + contributions[8];
                contributions[16] = contributions[16] + contributions[24];
                contributions[32] = contributions[32] + contributions[40];
                contributions[0] = contributions[0] + contributions[16];
                contributions[0] = contributions[0] + contributions[32];
                float result_0 = contributions[0];
                if (payload_dtype_code == 1) {
                    carrier_values[0] = (float)(__half)result_0;
                } else {
                    carrier_values[0] = (float)(__nv_bfloat16)result_0;
                }
                contributions[1] = contributions[1] + contributions[3];
                contributions[5] = contributions[5] + contributions[7];
                contributions[9] = contributions[9] + contributions[11];
                contributions[13] = contributions[13] + contributions[15];
                contributions[17] = contributions[17] + contributions[19];
                contributions[21] = contributions[21] + contributions[23];
                contributions[25] = contributions[25] + contributions[27];
                contributions[29] = contributions[29] + contributions[31];
                contributions[33] = contributions[33] + contributions[35];
                contributions[37] = contributions[37] + contributions[39];
                contributions[41] = contributions[41] + contributions[43];
                contributions[1] = contributions[1] + contributions[5];
                contributions[9] = contributions[9] + contributions[13];
                contributions[17] = contributions[17] + contributions[21];
                contributions[25] = contributions[25] + contributions[29];
                contributions[33] = contributions[33] + contributions[37];
                contributions[1] = contributions[1] + contributions[9];
                contributions[17] = contributions[17] + contributions[25];
                contributions[33] = contributions[33] + contributions[41];
                contributions[1] = contributions[1] + contributions[17];
                contributions[1] = contributions[1] + contributions[33];
                float result_1 = contributions[1];
                if (payload_dtype_code == 1) {
                    carrier_values[1] = (float)(__half)result_1;
                } else {
                    carrier_values[1] = (float)(__nv_bfloat16)result_1;
                }
            }
            int block_column = column / block_size;
            float block_max;
            asm("max.f32 %0, %1, %2;" : "=f"(block_max) : "f"(fabsf(carrier_values[0])), "f"(fabsf(carrier_values[1])));
            if (quant_mode == 3) {
                block_max = max_noftz(block_max, 0.0f);
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_3);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_5);
            } else {
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
                block_max = max_noftz(block_max, _shfl_xor_sync_6);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
                block_max = max_noftz(block_max, _shfl_xor_sync_7);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
                block_max = max_noftz(block_max, _shfl_xor_sync_8);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
                block_max = max_noftz(block_max, _shfl_xor_sync_9);
            }
            int scale_byte = 0;
            float actual_scale = 0.0f;
            float fp8_scale = 0.0f;
            if (quant_mode == 3) {
                float reciprocal_six = approx_rcp(6.0f);
                float sf_value = output_scalar_scale * (block_max * reciprocal_six);
                uint16_t _e4m3x2_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
                uint32_t _f16x2_decode_13;
                float _fp8_e4m3_decode_10;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
                fp8_scale = _fp8_e4m3_decode_10;
                if (block_max != 0.0f) {
                    float reciprocal_scalar = approx_rcp(output_scalar_scale);
                    float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
                    actual_scale = _rcp_14;
                }
            } else {
                float denominator = 6.0f;
                if (quant_mode == 1) {
                    denominator = 448.0f;
                }
                float _rcp_15 = approx_rcp(denominator);
                float raw_scale = block_max * _rcp_15;
                int raw_bits = 0;
                raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
                int exponent = raw_bits >> 23 & 255;
                int mantissa = raw_bits & 8388607;
                int round_up = 0;
                if (mantissa != 0) {
                    if (exponent != 0) {
                        round_up = 1;
                    } else if (mantissa > 4194304) {
                        round_up = 1;
                    }
                }
                if (raw_scale > 0.0f) {
                    scale_byte = exponent + round_up;
                    if (scale_byte > 254) {
                        scale_byte = 254;
                    }
                }
                if (quant_mode == 1) {
                    int scale_bits = scale_byte << 23;
                    if (scale_byte == 0) {
                        scale_bits = 4194304;
                    }
                    float decoded_scale = 0.0f;
                    decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
                    float _rcp_16 = approx_rcp(decoded_scale);
                    actual_scale = _rcp_16;
                } else if (block_max != 0.0f) {
                    if (scale_byte == 0) {
                        actual_scale = 1.0f;
                    } else {
                        float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                        actual_scale = _exp2_17;
                    }
                }
            }
            int scale_index = token * blocks_per_row + block_column;
            if (scale_layout == 0) {
                scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
            }
            if (scale_layout == 1) {
                int tiles = padded_scale_cols / 4;
                scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
            }
            int block_lane = lane & 15;
            if (quant_mode == 3) {
                block_lane = lane & 7;
            }
            if (active != 0) {
                if (block_lane == 0) {
                    if (quant_mode == 3) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_18;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                        *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
                    } else {
                        scales_u8[scale_index] = scale_byte;
                    }
                }
                float normalized_low = carrier_values[0] * actual_scale;
                float normalized_high = carrier_values[1] * actual_scale;
                if (quant_mode == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(normalized_high), "f"(normalized_low));
                    *reinterpret_cast<unsigned short*>(quantized_fp8 + (token * elements_per_token + column)) = (unsigned short)(_cvt_satfinite_e4m3x2_f32_19);
                } else {
                    uint32_t _fp4_e2m1_pack_pair_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_e2m1_pack_pair_20) : "f"(normalized_low), "f"(normalized_high));
                    quantized_packed[token * (elements_per_token / 2) + column / 2] = _fp4_e2m1_pack_pair_20;
                }
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#undef TOP_K

extern "C" {

__global__ __launch_bounds__(128) void
kernel_flashinfer_mnnvl_moe_alltoall_quantize_combine(uint8_t* __restrict__ accumulated, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, int tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int block_size = 32;
    int chunk_shift = 1;
    if (quant_mode == 3) {
        block_size = 16;
        chunk_shift = 0;
    }
    int chunk = bid * 128 + tid;
    int logical_block = chunk >> chunk_shift;
    int block_chunk = chunk - (logical_block << chunk_shift);
    int token = logical_block / blocks_per_row;
    int block_column = logical_block - token * blocks_per_row;
    int column = block_column * block_size + block_chunk * 16;
    int active = ((token < tokens) ? 1 : 0);
    float values[16];
    #pragma unroll
    for (int element = 0; element < 16; element++) {
        values[element] = 0.0f;
    }
    if (active != 0) {
        unsigned long long accumulated_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
        if (payload_dtype_code == 0) {
            __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(accumulated);
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(source_bf16 + accumulated_item * (unsigned long long)payload_element_bytes / 2ull);
                uint4 _vld_0[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&values[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                        (&values[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                    }
                }
            }
        }
        if (payload_dtype_code == 1) {
            __half* source_f16 = reinterpret_cast<__half*>(accumulated);
            unsigned long long f16_index = accumulated_item * (unsigned long long)payload_element_bytes / 2ull;
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(source_f16 + f16_index);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + _blk * 8 + _pair * 2]))
                            : "r"(_vpairs_1[_pair]));
                    }
                }
            }
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(source_f16 + f16_index + 8ull);
                uint4 _vld_2[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&values[8 + _blk * 8 + _pair * 2]))
                            : "r"(_vpairs_2[_pair]));
                    }
                }
            }
        }
        if (payload_dtype_code == 2) {
            uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(accumulated);
            unsigned long long fp8_index = accumulated_item * (unsigned long long)payload_element_bytes;
            {
                uint64_t _fp8x8_3 = *reinterpret_cast<const uint64_t*>(source_fp8 + fp8_index);
                uint16_t _e4m3x2_0_3 = (uint16_t)((_fp8x8_3 >> 0) & 0xFFFFull);
                uint32_t _f16x2_0_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_3) : "h"(_e4m3x2_0_3));
                uint16_t _h0_3 = (uint16_t)((_f16x2_0_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 0]) : "h"(_h0_3));
                uint16_t _h1_3 = (uint16_t)((_f16x2_0_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 1]) : "h"(_h1_3));
                uint16_t _e4m3x2_1_3 = (uint16_t)((_fp8x8_3 >> 16) & 0xFFFFull);
                uint32_t _f16x2_1_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_3) : "h"(_e4m3x2_1_3));
                uint16_t _h2_3 = (uint16_t)((_f16x2_1_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 2]) : "h"(_h2_3));
                uint16_t _h3_3 = (uint16_t)((_f16x2_1_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 3]) : "h"(_h3_3));
                uint16_t _e4m3x2_2_3 = (uint16_t)((_fp8x8_3 >> 32) & 0xFFFFull);
                uint32_t _f16x2_2_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_3) : "h"(_e4m3x2_2_3));
                uint16_t _h4_3 = (uint16_t)((_f16x2_2_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 4]) : "h"(_h4_3));
                uint16_t _h5_3 = (uint16_t)((_f16x2_2_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 5]) : "h"(_h5_3));
                uint16_t _e4m3x2_3_3 = (uint16_t)((_fp8x8_3 >> 48) & 0xFFFFull);
                uint32_t _f16x2_3_3;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_3) : "h"(_e4m3x2_3_3));
                uint16_t _h6_3 = (uint16_t)((_f16x2_3_3 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 6]) : "h"(_h6_3));
                uint16_t _h7_3 = (uint16_t)((_f16x2_3_3 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[0 + 7]) : "h"(_h7_3));
            }
            {
                uint64_t _fp8x8_4 = *reinterpret_cast<const uint64_t*>(source_fp8 + fp8_index + 8ull);
                uint16_t _e4m3x2_0_4 = (uint16_t)((_fp8x8_4 >> 0) & 0xFFFFull);
                uint32_t _f16x2_0_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0_4) : "h"(_e4m3x2_0_4));
                uint16_t _h0_4 = (uint16_t)((_f16x2_0_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[8 + 0]) : "h"(_h0_4));
                uint16_t _h1_4 = (uint16_t)((_f16x2_0_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[8 + 1]) : "h"(_h1_4));
                uint16_t _e4m3x2_1_4 = (uint16_t)((_fp8x8_4 >> 16) & 0xFFFFull);
                uint32_t _f16x2_1_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1_4) : "h"(_e4m3x2_1_4));
                uint16_t _h2_4 = (uint16_t)((_f16x2_1_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[8 + 2]) : "h"(_h2_4));
                uint16_t _h3_4 = (uint16_t)((_f16x2_1_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[8 + 3]) : "h"(_h3_4));
                uint16_t _e4m3x2_2_4 = (uint16_t)((_fp8x8_4 >> 32) & 0xFFFFull);
                uint32_t _f16x2_2_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2_4) : "h"(_e4m3x2_2_4));
                uint16_t _h4_4 = (uint16_t)((_f16x2_2_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[8 + 4]) : "h"(_h4_4));
                uint16_t _h5_4 = (uint16_t)((_f16x2_2_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[8 + 5]) : "h"(_h5_4));
                uint16_t _e4m3x2_3_4 = (uint16_t)((_fp8x8_4 >> 48) & 0xFFFFull);
                uint32_t _f16x2_3_4;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3_4) : "h"(_e4m3x2_3_4));
                uint16_t _h6_4 = (uint16_t)((_f16x2_3_4 >> 0) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[8 + 6]) : "h"(_h6_4));
                uint16_t _h7_4 = (uint16_t)((_f16x2_3_4 >> 16) & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(values[8 + 7]) : "h"(_h7_4));
            }
        }
        if (payload_dtype_code == 3) {
            float* source_f32 = reinterpret_cast<float*>(accumulated);
            unsigned long long f32_index = accumulated_item * (unsigned long long)payload_element_bytes / 4ull;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(source_f32 + f32_index);
                values[0 + 0] = _v4.x;
                values[0 + 1] = _v4.y;
                values[0 + 2] = _v4.z;
                values[0 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(source_f32 + f32_index + 4ull);
                values[4 + 0] = _v4.x;
                values[4 + 1] = _v4.y;
                values[4 + 2] = _v4.z;
                values[4 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(source_f32 + f32_index + 8ull);
                values[8 + 0] = _v4.x;
                values[8 + 1] = _v4.y;
                values[8 + 2] = _v4.z;
                values[8 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(source_f32 + f32_index + 12ull);
                values[12 + 0] = _v4.x;
                values[12 + 1] = _v4.y;
                values[12 + 2] = _v4.z;
                values[12 + 3] = _v4.w;
            }
        }
    }
    float block_max = fabsf(values[0]);
    #pragma unroll
    for (int element_1 = 1; element_1 < 16; element_1++) {
        block_max = max_noftz(block_max, fabsf(values[element_1]));
    }
    if (quant_mode == 3) {
        block_max = max_noftz(block_max, 0.0f);
    } else {
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
        block_max = max_noftz(block_max, _shfl_xor_sync_9);
    }
    int scale_byte = 0;
    float actual_scale = 0.0f;
    float fp8_scale = 0.0f;
    if (quant_mode == 3) {
        float reciprocal_six = approx_rcp(6.0f);
        float sf_value = output_scalar_scale * (block_max * reciprocal_six);
        uint16_t _e4m3x2_12;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
        uint8_t _cvt_satfinite_e4m3_f32_11 = (uint8_t)_e4m3x2_12;
        uint32_t _f16x2_decode_13;
        float _fp8_e4m3_decode_10;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_12));
        uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_e4m3_decode_10) : "h"(_f16_decode_13));
        fp8_scale = _fp8_e4m3_decode_10;
        if (block_max != 0.0f) {
            float reciprocal_scalar = approx_rcp(output_scalar_scale);
            float _rcp_14 = approx_rcp(fp8_scale * reciprocal_scalar);
            actual_scale = _rcp_14;
        }
    } else {
        float denominator = 6.0f;
        if (quant_mode == 1) {
            denominator = 448.0f;
        }
        float _rcp_15 = approx_rcp(denominator);
        float raw_scale = block_max * _rcp_15;
        int raw_bits = 0;
        raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
        int exponent = raw_bits >> 23 & 255;
        int mantissa = raw_bits & 8388607;
        int round_up = 0;
        if (mantissa != 0) {
            if (exponent != 0) {
                round_up = 1;
            } else if (mantissa > 4194304) {
                round_up = 1;
            }
        }
        if (raw_scale > 0.0f) {
            scale_byte = exponent + round_up;
            if (scale_byte > 254) {
                scale_byte = 254;
            }
        }
        if (quant_mode == 1) {
            int scale_bits = scale_byte << 23;
            if (scale_byte == 0) {
                scale_bits = 4194304;
            }
            float decoded_scale = 0.0f;
            decoded_scale = reinterpret_cast<float*>(&scale_bits)[0];
            float _rcp_16 = approx_rcp(decoded_scale);
            actual_scale = _rcp_16;
        } else if (block_max != 0.0f) {
            if (scale_byte == 0) {
                actual_scale = 1.0f;
            } else {
                float _exp2_17 = approx_exp2(127.0f - (float)scale_byte);
                actual_scale = _exp2_17;
            }
        }
    }
    int scale_index = token * blocks_per_row + block_column;
    if (scale_layout == 0) {
        scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
    }
    if (scale_layout == 1) {
        int tiles = padded_scale_cols / 4;
        scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
    }
    if (active != 0) {
        if (block_chunk == 0) {
            if (quant_mode == 3) {
                uint16_t _cvt_satfinite_e4m3x2_f32_18;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(0.0f), "f"(fp8_scale));
                *reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_18));
            } else {
                scales_u8[scale_index] = scale_byte;
            }
        }
        float normalized[16];
        #pragma unroll
        for (int element_2 = 0; element_2 < 16; element_2++) {
            normalized[element_2] = values[element_2] * actual_scale;
        }
        if (quant_mode == 1) {
            unsigned int fp8_words[4];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(normalized[0]), "f"(normalized[1]),
                                       "f"(normalized[2]), "f"(normalized[3]));
                fp8_words[0] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(normalized[4]), "f"(normalized[5]),
                                       "f"(normalized[6]), "f"(normalized[7]));
                fp8_words[1] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(normalized[8]), "f"(normalized[9]),
                                       "f"(normalized[10]), "f"(normalized[11]));
                fp8_words[2] = _packed;
            }
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(normalized[12]), "f"(normalized[13]),
                                       "f"(normalized[14]), "f"(normalized[15]));
                fp8_words[3] = _packed;
            }
            reinterpret_cast<int4*>(quantized_fp8 + (token * elements_per_token + column))[0] = reinterpret_cast<int4*>(fp8_words)[0];
        } else {
            unsigned int fp4_words[2];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(fp4_words[0]) : "f"(normalized[0]), "f"(normalized[1]), "f"(normalized[2]), "f"(normalized[3]), "f"(normalized[4]), "f"(normalized[5]), "f"(normalized[6]), "f"(normalized[7]));
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(fp4_words[1]) : "f"(normalized[8]), "f"(normalized[9]), "f"(normalized[10]), "f"(normalized[11]), "f"(normalized[12]), "f"(normalized[13]), "f"(normalized[14]), "f"(normalized[15]));
            reinterpret_cast<int2*>(quantized_packed + (token * (elements_per_token / 2) + column / 2))[0] = reinterpret_cast<int2*>(fp4_words)[0];
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_flashinfer_mnnvl_moe_alltoall_sanitize_expert_ids(int* __restrict__ expert_ids, int* __restrict__ recv_counters, int ep_size, int max_tokens_per_rank, int top_k, int invalid_id, int enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    if (enable_pdl != 0) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int tid_0 = bid * 256 + tid;
    int total_tokens = ep_size * max_tokens_per_rank;
    if (tid_0 < total_tokens) {
        int source_rank = tid_0 / max_tokens_per_rank;
        int token_idx = tid_0 - source_rank * max_tokens_per_rank;
        if (token_idx >= recv_counters[source_rank]) {
            int token_base = tid_0 * top_k;
            #pragma unroll 1
            for (int k = 0; k < top_k; k++) {
                expert_ids[token_base + k] = invalid_id;
            }
        }
    }
}

} // extern "C"

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef THREADS
