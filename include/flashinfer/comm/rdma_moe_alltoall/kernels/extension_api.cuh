/*
 * Derived from DeepEP (Copyright (c) 2025 DeepSeek), MIT licensed.
 * Copyright (c) 2026 by FlashInfer team.
 * See LICENSE.deepep.txt and include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md.
 */
#pragma once

#include <vector>
#ifdef DEEP_EP_USE_GIN
#include "gin_device.cuh"
#define EP_GIN_HOST_ARG , const deep_ep::GinLaunch& gin_launch
#else
#define EP_GIN_HOST_ARG
#endif

namespace deep_ep {
namespace extensions {
void quantize_bf16_to_nvfp4(const void* input, const float* global_scale, void* nvfp4_packed_output,
                            void* fp8_scales, int token_num, int hidden_dim, cudaStream_t stream);
void quantize_bf16_to_nvfp4_uniform(const void* input, float global_scale,
                                    void* nvfp4_packed_output, void* fp8_scales, int token_num,
                                    int hidden_dim, cudaStream_t stream);
void dequantize_nvfp4_to_bf16(const void* nvfp4_packed_input, const float* global_scale,
                              const void* fp8_scales, void* output, int token_num, int hidden_dim,
                              cudaStream_t stream);
void quantize_bf16_to_fp8(const void* input, void* fp8_output, void* fp32_scales, int token_num,
                          int hidden_dim, cudaStream_t stream);
void dequantize_fp8_to_bf16(const void* fp8_input, const void* fp32_scales, void* output,
                            int token_num, int hidden_dim, cudaStream_t stream);
// Precision IDs are kept compatible with combine: FP8=0, NVFP4=1, BF16=2.
// Dispatch transports prequantized values/scales; BF16 uses null scales.
#ifdef DEEP_EP_USE_GIN
void finalize_planned_dispatch(int tokens, int topk, int capacity, int ranks, int local_experts,
                               cudaStream_t stream, const GinLaunch& launch);
#endif
void dispatch(int precision, void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range, int* packed_recv_count,
              int* cumulative_local_expert_recv_stats, void* rdma_recv_x, int* rdma_recv_count,
              void* rdma_x, const void* x, const void* x_scales, const int* topk_idx,
              int* next_clean, int num_next_clean_int, int num_tokens, int hidden,
              int num_max_dispatch_tokens_per_rank, int num_topk, int num_experts, int rank,
              int num_ranks, void* workspace, int num_device_sms, cudaStream_t stream,
              int phases EP_GIN_HOST_ARG);
void low_precision_combine(int precision, void* combined_x, void* rdma_recv_x, int* rdma_recv_flag,
                           void* rdma_send_x, const void* x, const float* global_scale_per_token,
                           const int* topk_idx, const float* topk_weights, const int* src_info,
                           const int64_t* layout_range, int* next_clean, int num_next_clean_int,
                           int num_combined_tokens, int hidden, int num_topk,
                           int num_max_dispatch_tokens_per_rank, int num_experts, int rank,
                           int num_ranks, void* workspace, int num_device_sms, cudaStream_t stream,
                           int phases EP_GIN_HOST_ARG);
}  // namespace extensions
}  // namespace deep_ep
