/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_TARGET_RANKS_OFF 0
#define SMEM_SMEM_TARGET_RANKS_STAGE_BYTES 88
#define SMEM_SMEM_TARGET_RANKS_STRIDE 88
#define SMEM_SMEM_SEND_INDICES_OFF 88
#define SMEM_SMEM_SEND_INDICES_STAGE_BYTES 88
#define SMEM_SMEM_SEND_INDICES_STRIDE 88
#define SMEM_TOTAL 256
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_dispatch(int* __restrict__ token_selected_experts, uint8_t* __restrict__ payload_0, uint8_t* __restrict__ payload_1, uint8_t* __restrict__ payload_2, uint8_t* __restrict__ payload_3, uint8_t* __restrict__ payload_4, uint8_t* __restrict__ payload_5, uint8_t* __restrict__ workspace, int* __restrict__ eplb_local_stats, unsigned long long workspace_stride_bytes, unsigned long long flag_val_offset, unsigned long long local_token_counter_offset, unsigned long long send_counters_offset, unsigned long long recv_counters_offset, unsigned long long completion_flags_offset, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long eplb_gathered_stats_offset, unsigned long long payload_0_offset, unsigned long long payload_1_offset, unsigned long long payload_2_offset, unsigned long long payload_3_offset, unsigned long long payload_4_offset, unsigned long long payload_5_offset, int payload_0_bytes, int payload_1_bytes, int payload_2_bytes, int payload_3_bytes, int payload_4_bytes, int payload_5_bytes, int num_payloads, int max_tokens_per_rank, int local_num_tokens, int ep_rank, int ep_size, int num_experts, int top_k, int eplb_stats_num_experts, bool enable_pdl, bool enable_eplb, bool enable_rank_mask, unsigned long long active_rank_mask)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    int* smem_target_ranks = reinterpret_cast<int*>(smem_raw + 0);
    const int smem_target_ranks_addr = smem + 0;
    int* smem_send_indices = reinterpret_cast<int*>(smem_raw + 88);
    const int smem_send_indices_addr = smem + 88;

    // === Task calls (dependency order) ===
    int* workspace_i32 = reinterpret_cast<int*>(workspace);
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    int local_token_idx = bid;
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    if (local_num_tokens > 0) {
        if (tid == 0) {
            int ep_base = num_experts / ep_size;
            int ep_remainder = num_experts - ep_base * ep_size;
            int split = ep_remainder * (ep_base + 1);
            unsigned long long seen_ranks = 0;
            int route_base = local_token_idx * top_k;
            unsigned long long topk_workspace_base = (local_workspace_base + topk_target_ranks_offset) / 4 + (unsigned long long)local_token_idx * (unsigned long long)top_k;
            unsigned long long send_index_workspace_base = (local_workspace_base + topk_send_indices_offset) / 4 + (unsigned long long)local_token_idx * (unsigned long long)top_k;
            #pragma unroll 1
            for (int k = 0; k < top_k; k++) {
                int expert_id = token_selected_experts[route_base + k];
                int target_rank = 0;
                if (ep_remainder == 0) {
                    target_rank = expert_id / ep_base;
                } else if (expert_id < split) {
                    target_rank = expert_id / (ep_base + 1);
                } else {
                    target_rank = ep_remainder + (expert_id - split) / ep_base;
                }
                unsigned long long target_bit = 1ULL << (unsigned long long)target_rank;
                int first_for_rank = (((seen_ranks & target_bit) == 0) ? 1 : 0);
                seen_ranks = seen_ranks | target_bit;
                int rank_is_active = 1;
                if (enable_rank_mask) {
                    rank_is_active = (((active_rank_mask & target_bit) != 0) ? 1 : 0);
                }
                int stored_rank = -1;
                int send_index = -1;
                if (first_for_rank != 0 && rank_is_active != 0) {
                    unsigned long long send_counter_index = (local_workspace_base + send_counters_offset) / 4 + (unsigned long long)target_rank;
                    int _atomic_old_0 = atomicAdd(&workspace_i32[send_counter_index], 1);
                    send_index = _atomic_old_0;
                    stored_rank = target_rank;
                }
                smem_target_ranks[k] = stored_rank;
                smem_send_indices[k] = send_index;
                workspace_i32[topk_workspace_base + (unsigned long long)k] = stored_rank;
                workspace_i32[send_index_workspace_base + (unsigned long long)k] = send_index;
            }
        }
        __syncthreads();
        if (num_payloads > 0) {
            unsigned long long source_base_0 = (unsigned long long)local_token_idx * (unsigned long long)payload_0_bytes;
            #pragma unroll 1
            for (int byte_0 = tid; byte_0 < payload_0_bytes; byte_0 += 256) {
                uint8_t byte_value_0 = payload_0[source_base_0 + (unsigned long long)byte_0];
                #pragma unroll 1
                for (int k_0 = 0; k_0 < top_k; k_0++) {
                    int target_rank_0 = smem_target_ranks[k_0];
                    int send_index_0 = smem_send_indices[k_0];
                    if (send_index_0 >= 0) {
                        unsigned long long destination_0 = (unsigned long long)target_rank_0 * workspace_stride_bytes + payload_0_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_0) * (unsigned long long)payload_0_bytes + (unsigned long long)byte_0;
                        workspace[destination_0] = byte_value_0;
                    }
                }
            }
        }
        if (num_payloads > 1) {
            unsigned long long source_base_1 = (unsigned long long)local_token_idx * (unsigned long long)payload_1_bytes;
            #pragma unroll 1
            for (int byte_1 = tid; byte_1 < payload_1_bytes; byte_1 += 256) {
                uint8_t byte_value_1 = payload_1[source_base_1 + (unsigned long long)byte_1];
                #pragma unroll 1
                for (int k_1 = 0; k_1 < top_k; k_1++) {
                    int target_rank_1 = smem_target_ranks[k_1];
                    int send_index_1 = smem_send_indices[k_1];
                    if (send_index_1 >= 0) {
                        unsigned long long destination_1 = (unsigned long long)target_rank_1 * workspace_stride_bytes + payload_1_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_1) * (unsigned long long)payload_1_bytes + (unsigned long long)byte_1;
                        workspace[destination_1] = byte_value_1;
                    }
                }
            }
        }
        if (num_payloads > 2) {
            unsigned long long source_base_2 = (unsigned long long)local_token_idx * (unsigned long long)payload_2_bytes;
            #pragma unroll 1
            for (int byte_2 = tid; byte_2 < payload_2_bytes; byte_2 += 256) {
                uint8_t byte_value_2 = payload_2[source_base_2 + (unsigned long long)byte_2];
                #pragma unroll 1
                for (int k_2 = 0; k_2 < top_k; k_2++) {
                    int target_rank_2 = smem_target_ranks[k_2];
                    int send_index_2 = smem_send_indices[k_2];
                    if (send_index_2 >= 0) {
                        unsigned long long destination_2 = (unsigned long long)target_rank_2 * workspace_stride_bytes + payload_2_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_2) * (unsigned long long)payload_2_bytes + (unsigned long long)byte_2;
                        workspace[destination_2] = byte_value_2;
                    }
                }
            }
        }
        if (num_payloads > 3) {
            unsigned long long source_base_3 = (unsigned long long)local_token_idx * (unsigned long long)payload_3_bytes;
            #pragma unroll 1
            for (int byte_3 = tid; byte_3 < payload_3_bytes; byte_3 += 256) {
                uint8_t byte_value_3 = payload_3[source_base_3 + (unsigned long long)byte_3];
                #pragma unroll 1
                for (int k_3 = 0; k_3 < top_k; k_3++) {
                    int target_rank_3 = smem_target_ranks[k_3];
                    int send_index_3 = smem_send_indices[k_3];
                    if (send_index_3 >= 0) {
                        unsigned long long destination_3 = (unsigned long long)target_rank_3 * workspace_stride_bytes + payload_3_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_3) * (unsigned long long)payload_3_bytes + (unsigned long long)byte_3;
                        workspace[destination_3] = byte_value_3;
                    }
                }
            }
        }
        if (num_payloads > 4) {
            unsigned long long source_base_4 = (unsigned long long)local_token_idx * (unsigned long long)payload_4_bytes;
            #pragma unroll 1
            for (int byte_4 = tid; byte_4 < payload_4_bytes; byte_4 += 256) {
                uint8_t byte_value_4 = payload_4[source_base_4 + (unsigned long long)byte_4];
                #pragma unroll 1
                for (int k_4 = 0; k_4 < top_k; k_4++) {
                    int target_rank_4 = smem_target_ranks[k_4];
                    int send_index_4 = smem_send_indices[k_4];
                    if (send_index_4 >= 0) {
                        unsigned long long destination_4 = (unsigned long long)target_rank_4 * workspace_stride_bytes + payload_4_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_4) * (unsigned long long)payload_4_bytes + (unsigned long long)byte_4;
                        workspace[destination_4] = byte_value_4;
                    }
                }
            }
        }
        if (num_payloads > 5) {
            unsigned long long source_base_5 = (unsigned long long)local_token_idx * (unsigned long long)payload_5_bytes;
            #pragma unroll 1
            for (int byte_5 = tid; byte_5 < payload_5_bytes; byte_5 += 256) {
                uint8_t byte_value_5 = payload_5[source_base_5 + (unsigned long long)byte_5];
                #pragma unroll 1
                for (int k_5 = 0; k_5 < top_k; k_5++) {
                    int target_rank_5 = smem_target_ranks[k_5];
                    int send_index_5 = smem_send_indices[k_5];
                    if (send_index_5 >= 0) {
                        unsigned long long destination_5 = (unsigned long long)target_rank_5 * workspace_stride_bytes + payload_5_offset + ((unsigned long long)ep_rank * (unsigned long long)max_tokens_per_rank + (unsigned long long)send_index_5) * (unsigned long long)payload_5_bytes + (unsigned long long)byte_5;
                        workspace[destination_5] = byte_value_5;
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
                int _atomic_old_1 = atomicAdd(&workspace_i32[local_counter_index], 1);
                int completed_before = _atomic_old_1;
                if (completed_before + 1 == local_num_tokens) {
                    is_last_token = 1;
                }
            }
        }
        int _shfl_0 = __shfl_sync(0xFFFFFFFF, is_last_token, 0);
        is_last_token = _shfl_0;
        if (is_last_token != 0) {
            #pragma unroll 1
            for (int target_rank_6 = lane; target_rank_6 < ep_size; target_rank_6 += 32) {
                int rank_is_active_1 = 1;
                if (enable_rank_mask) {
                    unsigned long long target_bit_1 = 1ULL << (unsigned long long)target_rank_6;
                    rank_is_active_1 = (((active_rank_mask & target_bit_1) != 0) ? 1 : 0);
                }
                if (rank_is_active_1 != 0) {
                    unsigned long long send_counter_index_1 = (local_workspace_base + send_counters_offset) / 4 + (unsigned long long)target_rank_6;
                    unsigned long long recv_counter_index = ((unsigned long long)target_rank_6 * workspace_stride_bytes + recv_counters_offset) / 4 + (unsigned long long)ep_rank;
                    workspace_i32[recv_counter_index] = workspace_i32[send_counter_index_1];
                }
            }
            if (enable_eplb) {
                #pragma unroll 1
                for (int target_rank_eplb = 0; target_rank_eplb < ep_size; target_rank_eplb++) {
                    int rank_is_active_eplb = 1;
                    if (enable_rank_mask) {
                        unsigned long long target_bit_eplb = 1ULL << (unsigned long long)target_rank_eplb;
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
                    unsigned long long target_bit_flag = 1ULL << (unsigned long long)target_rank_flag;
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
                    unsigned long long peer_bit = 1ULL << (unsigned long long)peer_rank;
                    peer_is_active = (((active_rank_mask & peer_bit) != 0) ? 1 : 0);
                }
                if (peer_is_active != 0) {
                    unsigned long long local_flag_index = (local_workspace_base + completion_flags_offset) / 4 + (unsigned long long)peer_rank;
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
        }
    }
}

} // extern "C"
