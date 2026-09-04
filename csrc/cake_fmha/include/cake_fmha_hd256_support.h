/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone support ABI for Cake's staged head-dim-256 context routes.
 * Product minimum architecture: sm_100a.
 */

#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

extern "C" cudaError_t cake_fmha_launch_hd256_stage_q(
    const uint8_t* q, uint8_t* q_packed, const int32_t* q_indptr, int batch_size, int num_q_heads,
    int padded_q, int head_dim_bytes, int64_t q_token_stride_bytes, int64_t q_head_stride_bytes,
    unsigned int grid_x, cudaStream_t stream);

extern "C" cudaError_t cake_fmha_launch_hd256_stage_kv(
    const uint8_t* k_source, const uint8_t* v_source, uint8_t* k_packed, uint8_t* v_packed,
    const int32_t* page_table, const int32_t* seq_lens, int batch_size, int num_kv_heads,
    int page_size, int max_micro_pages, int head_dim_bytes, int64_t source_page_stride_bytes,
    int64_t source_token_stride_bytes, int64_t source_head_stride_bytes,
    int64_t page_table_batch_stride, int64_t page_table_side_stride, unsigned int grid_x,
    cudaStream_t stream);

extern "C" cudaError_t cake_fmha_launch_hd256_prepare_metadata(
    const int32_t* q_indptr, const int32_t* seq_lens, int32_t* seq_lens_q, int32_t* seq_lens_kv,
    int32_t* cu_seq_lens_q, int32_t* kernel_page_table, uint32_t* dynamic_counter, int batch_size,
    int num_q_heads, int num_kv_heads, int padded_q, int max_micro_pages, unsigned int grid_x,
    cudaStream_t stream);

extern "C" cudaError_t cake_fmha_launch_hd256_scatter_o(
    const uint8_t* o_packed, uint8_t* output, const int32_t* q_indptr, int batch_size,
    int num_q_heads, int padded_q, int head_dim_bytes, int64_t output_token_stride_bytes,
    int64_t output_head_stride_bytes, unsigned int grid_x, cudaStream_t stream);
