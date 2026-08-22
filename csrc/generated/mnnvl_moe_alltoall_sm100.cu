// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Generated source. Do not edit.
#include <stdint.h>
#include <cuda.h>
#include <cuda_bf16.h>

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

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 64

extern "C" {

__global__ __launch_bounds__(64) void
kernel_flashinfer_mnnvl_moe_alltoall_prepare_dispatch(int* __restrict__ send_counters, int* __restrict__ local_token_counter, int ep_size, unsigned int* __restrict__ flag_val, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
#define SMEM_TOTAL 256
#define THREADS 256

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
                unsigned long long rank_bit_one = 1;
                unsigned long long target_bit = rank_bit_one << (unsigned long long)target_rank;
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
                    unsigned long long completion_bit_one = 1;
                    unsigned long long target_bit_1 = completion_bit_one << (unsigned long long)target_rank_6;
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

#undef FLASHINFER_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_SEND_INDICES_OFF
#undef SMEM_SMEM_SEND_INDICES_STAGE_BYTES
#undef SMEM_SMEM_SEND_INDICES_STRIDE
#undef SMEM_SMEM_TARGET_RANKS_OFF
#undef SMEM_SMEM_TARGET_RANKS_STAGE_BYTES
#undef SMEM_SMEM_TARGET_RANKS_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef smem_send_indices_addr
#undef smem_target_ranks_addr

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_stage_combine(uint8_t* __restrict__ payload, uint8_t* __restrict__ workspace, unsigned long long workspace_stride_bytes, unsigned long long flag_val_offset, unsigned long long destination_offset, unsigned long long payload_bytes, int ep_rank, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (bid == 0) {
        if (tid == 0) {
            unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
            unsigned long long flag_index = ((unsigned long long)ep_rank * workspace_stride_bytes + flag_val_offset) / 4;
            workspace_u32[flag_index] = workspace_u32[flag_index] + 1;
        }
    }
    unsigned long long linear_thread = (unsigned long long)bid * 256 + (unsigned long long)tid;
    unsigned long long thread_stride = (unsigned long long)num_bids * 256;
    #pragma unroll 1
    for (unsigned long long byte_index = linear_thread; byte_index < payload_bytes; byte_index += thread_stride) {
        workspace[destination_offset + byte_index] = payload[byte_index];
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

__global__ __launch_bounds__(64) void
kernel_flashinfer_mnnvl_moe_alltoall_publish_combine(uint8_t* __restrict__ workspace, unsigned long long workspace_stride_bytes, unsigned long long flag_val_offset, unsigned long long completion_flags_offset, int ep_rank, int ep_size, bool enable_pdl, bool enable_rank_mask, unsigned long long active_rank_mask)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    unsigned int* workspace_u32 = reinterpret_cast<unsigned int*>(workspace);
    unsigned long long local_workspace_base = (unsigned long long)ep_rank * workspace_stride_bytes;
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    unsigned long long expected_index = (local_workspace_base + flag_val_offset) / 4;
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

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_1(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
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
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                {
                    {
                        {
                            {
                                {
                                    {
                                    }
                                }
                            }
                        }
                    }
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

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_2(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
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
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                {
                    {
                        {
                            {
                                {
                                    {
                                        contributions[0] = contributions[0] + contributions[1];
                                    }
                                }
                            }
                        }
                    }
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
#define TOP_K 4

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_4(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
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
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                {
                    {
                        {
                            {
                                {
                                    contributions[0] = contributions[0] + contributions[1];
                                    contributions[2] = contributions[2] + contributions[3];
                                    contributions[0] = contributions[0] + contributions[2];
                                }
                            }
                        }
                    }
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
#define TOP_K 6

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_6(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
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
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                {
                    {
                        {
                            {
                                contributions[0] = contributions[0] + contributions[1];
                                contributions[2] = contributions[2] + contributions[3];
                                contributions[4] = contributions[4] + contributions[5];
                                contributions[0] = contributions[0] + contributions[2];
                                contributions[0] = contributions[0] + contributions[4];
                            }
                        }
                    }
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
#define TOP_K 8

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_8(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
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
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                {
                    {
                        {
                            contributions[0] = contributions[0] + contributions[1];
                            contributions[2] = contributions[2] + contributions[3];
                            contributions[4] = contributions[4] + contributions[5];
                            contributions[6] = contributions[6] + contributions[7];
                            contributions[0] = contributions[0] + contributions[2];
                            contributions[4] = contributions[4] + contributions[6];
                            contributions[0] = contributions[0] + contributions[4];
                        }
                    }
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
#define TOP_K 10

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_10(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
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
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                {
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
#define TOP_K 12

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_12(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
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
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                {
                    {
                        {
                            {
                                {
                                    {
                                        {
                                            #pragma unroll
                                            for (int route_reduce = 1; route_reduce < TOP_K; route_reduce++) {
                                                contributions[0] = contributions[0] + contributions[route_reduce];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
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

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_14(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
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
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                {
                    {
                        {
                            {
                                {
                                    {
                                        {
                                            #pragma unroll
                                            for (int route_reduce = 1; route_reduce < TOP_K; route_reduce++) {
                                                contributions[0] = contributions[0] + contributions[route_reduce];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
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

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_16(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
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
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
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

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_18(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
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
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
                        contribution = _fp8_rt_0;
                    }
                    contributions[route] = contribution;
                }
            }
            {
                {
                    {
                        {
                            {
                                {
                                    {
                                        {
                                            #pragma unroll
                                            for (int route_reduce = 1; route_reduce < TOP_K; route_reduce++) {
                                                contributions[0] = contributions[0] + contributions[route_reduce];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
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

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_22(uint8_t* __restrict__ workspace, uint8_t* __restrict__ output, unsigned long long workspace_stride_bytes, unsigned long long topk_target_ranks_offset, unsigned long long topk_send_indices_offset, unsigned long long combine_payload_offset, int max_tokens_per_rank, int local_num_tokens, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int output_dtype_code, int ep_rank, bool use_low_precision, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
        #pragma unroll 1
        for (int column = tid; column < elements_per_token; column += 256) {
            float contributions[TOP_K];
            #pragma unroll
            for (int route_init = 0; route_init < TOP_K; route_init++) {
                contributions[route_init] = 0.0f;
            }
            #pragma unroll
            for (int route = 0; route < TOP_K; route++) {
                int target_rank = workspace_i32[target_base + (unsigned long long)route];
                int send_index = workspace_i32[send_base + (unsigned long long)route];
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
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(contribution));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
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

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32

extern "C" {

__global__ __launch_bounds__(32) void
kernel_flashinfer_mnnvl_moe_alltoall_quantize_combine(uint8_t* __restrict__ accumulated, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, int elements_per_token, int payload_element_bytes, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int logical_block = bid;
    int token = logical_block / blocks_per_row;
    int block_column = logical_block - token * blocks_per_row;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    int column = block_column * block_size + lane;
    int active_lane = ((block_size > lane) ? 1 : 0);
    float value = 0.0f;
    if (active_lane != 0) {
        unsigned long long accumulated_item = (unsigned long long)token * (unsigned long long)elements_per_token + (unsigned long long)column;
        float value_0 = 0.0f;
        if (payload_dtype_code == 0) {
            __nv_bfloat16* source_bf16 = reinterpret_cast<__nv_bfloat16*>(accumulated);
            value_0 = (float)source_bf16[accumulated_item * (unsigned long long)payload_element_bytes / 2];
        }
        if (payload_dtype_code == 1) {
            __half* source_f16 = reinterpret_cast<__half*>(accumulated);
            value_0 = (float)source_f16[accumulated_item * (unsigned long long)payload_element_bytes / 2];
        }
        if (payload_dtype_code == 2) {
            uint8_t* source_fp8 = reinterpret_cast<uint8_t*>(accumulated);
            float _vec_load_0[1];
            {
                uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(source_fp8 + accumulated_item * (unsigned long long)payload_element_bytes);
                uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                uint32_t _f16x2_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_0[0]) : "h"(_h0_0));
            }
            value_0 = _vec_load_0[0];
        }
        if (payload_dtype_code == 3) {
            float* source_f32 = reinterpret_cast<float*>(accumulated);
            value_0 = source_f32[accumulated_item * (unsigned long long)payload_element_bytes / 4];
        }
        value = value_0;
    }
    float scaled_value = value;
    float _fabs_0 = fabsf(scaled_value);
    float block_max = _fabs_0;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, block_max, 16);
    float _max_0 = max_noftz(block_max, _shfl_xor_0);
    block_max = _max_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
    float _max_1 = max_noftz(block_max, _shfl_xor_1);
    block_max = _max_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
    float _max_2 = max_noftz(block_max, _shfl_xor_2);
    block_max = _max_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
    float _max_3 = max_noftz(block_max, _shfl_xor_3);
    block_max = _max_3;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
    float _max_4 = max_noftz(block_max, _shfl_xor_4);
    block_max = _max_4;
    float _shfl_0 = __shfl_sync(0xFFFFFFFF, block_max, 0);
    block_max = _shfl_0;
    int scale_byte = 0;
    float actual_scale = 0.0f;
    float fp8_scale = 0.0f;
    if (quant_mode == 3) {
        float _rcp_0 = approx_rcp(6.0f);
        float reciprocal_six = _rcp_0;
        float sf_value = output_scalar_scale * (block_max * reciprocal_six);
        float _fp8_rt_0;
        uint16_t _e4m3x2_1;
        uint32_t _f16x2_1;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(sf_value));
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
        fp8_scale = _fp8_rt_0;
        if (block_max != 0.0f) {
            float _rcp_1 = approx_rcp(output_scalar_scale);
            float reciprocal_scalar = _rcp_1;
            float _rcp_2 = approx_rcp(fp8_scale * reciprocal_scalar);
            actual_scale = _rcp_2;
        }
    } else {
        float denominator = 6.0f;
        if (quant_mode == 1) {
            denominator = 448.0f;
        }
        float _rcp_3 = approx_rcp(denominator);
        float raw_scale = block_max * _rcp_3;
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
            float _rcp_4 = approx_rcp(decoded_scale);
            actual_scale = _rcp_4;
        } else if (block_max != 0.0f) {
            if (scale_byte == 0) {
                actual_scale = 1.0f;
            } else {
                float _cvt_f32_0 = __bfloat162float(scale_byte);
                float _exp2_0 = approx_exp2(127.0f - _cvt_f32_0);
                actual_scale = _exp2_0;
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
    if (lane == 0) {
        if (quant_mode == 3) {
            {
                unsigned short _fp8_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(fp8_scale));
                *(reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
            }
        } else {
            scales_u8[scale_index] = scale_byte;
        }
    }
    float normalized = 0.0f;
    if (quant_mode == 3) {
        normalized = value * actual_scale;
    } else {
        normalized = scaled_value * actual_scale;
    }
    if (quant_mode == 1) {
        if (active_lane != 0) {
            {
                unsigned short _fp8_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(normalized));
                *(reinterpret_cast<unsigned char*>(quantized_fp8 + (token * elements_per_token + column)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
            }
        }
    } else {
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, normalized, 1);
        float partner_value = _shfl_xor_5;
        uint32_t _fp4_pair_0;
        asm volatile("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_0) : "f"(normalized), "f"(partner_value));
        unsigned int packed_pair = _fp4_pair_0;
        if (active_lane != 0) {
            if ((lane & 1) == 0) {
                int packed_column = block_column * (block_size / 2) + lane / 2;
                quantized_packed[token * (elements_per_token / 2) + packed_column] = packed_pair;
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

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flashinfer_mnnvl_moe_alltoall_sanitize_expert_ids(int* __restrict__ expert_ids, int* __restrict__ recv_counters, int ep_size, int max_tokens_per_rank, int top_k, int invalid_id, int enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
