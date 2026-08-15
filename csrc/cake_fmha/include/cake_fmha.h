/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "cake_fmha_hd256_support.h"

extern "C" cudaError_t cake_fmha_launch_decode_native_bf16(
    const void* Qt,
    const void* K,
    const void* V,
    __nv_bfloat16* O_ptr,
    float* LSE_ptr,
    int* page_table,
    int* causal_seqlens_kv_global,
    float* scale_log2_ptr,
    float* sinks_ptr,
    int max_pages_per_seq,
    int max_local_seq_len,
    float softmax_scale_log2,
    int window_left,
    int num_q_heads,
    int num_kv_heads,
    int batch_size,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_decode_native_fp16_nhd(
    const void* Qt,
    const void* K,
    const void* V,
    __half* O_ptr,
    float* LSE_ptr,
    int* page_table,
    int* causal_seqlens_kv_global,
    float* scale_log2_ptr,
    float* sinks_ptr,
    int max_pages_per_seq,
    int max_local_seq_len,
    float softmax_scale_log2,
    int window_left,
    int num_q_heads,
    int num_kv_heads,
    int batch_size,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_decode_native_fp16_hd512(
    const void* Qt,
    const void* K,
    const void* V,
    __half* O_ptr,
    float* LSE_ptr,
    int* page_table,
    int* causal_seqlens_kv_global,
    float* scale_log2_ptr,
    int max_pages_per_seq,
    int max_local_seq_len,
    float softmax_scale_log2,
    int window_left,
    int num_q_heads,
    int num_kv_heads,
    int batch_size,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_decode_quant_fp8(
    const void* Qt,
    const void* K,
    const void* V,
    uint8_t* O,
    int* page_table,
    int* seq_lens_kv,
    float* bmm1_scale_ptr,
    float* bmm2_scale_ptr,
    float* partial_O,
    float* partial_max,
    float* partial_sum,
    int pt_batch_stride,
    int pt_v_offset,
    int bmm1_is_log2,
    int num_splits,
    int blocks_per_split,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_decode_quant_nvfp4(
    const void* Qt,
    const void* Kp,
    const void* Vp,
    const void* Ksf,
    const void* Vsf,
    uint8_t* O,
    int* page_table,
    int* seq_lens_kv,
    float* bmm1_scale_ptr,
    float* bmm2_scale_ptr,
    float* partial_O,
    float* partial_max,
    float* partial_sum,
    int pt_batch_stride,
    int pt_v_offset,
    int bmm1_is_log2,
    int num_splits,
    int blocks_per_split,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_decode_quant_bf16q(
    uint32_t* Q_ptr,
    const void* K,
    const void* V,
    __nv_bfloat16* O,
    int* page_table,
    int* seq_lens_kv,
    float* bmm1_scale_ptr,
    float* bmm2_scale_ptr,
    float* partial_O,
    float* partial_max,
    float* partial_sum,
    int pt_batch_stride,
    int pt_v_offset,
    int bmm1_is_log2,
    int num_splits,
    int blocks_per_split,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_decode_quant_fp8_reduce(
    float* partial_O,
    float* partial_max,
    float* partial_sum,
    uint8_t* O,
    float* bmm2_scale_ptr,
    int num_split,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_context_bf16(
    const void* Q,
    const void* K,
    const void* V,
    __nv_bfloat16* O_ptr,
    float* LSE_ptr,
    float* sinks,
    int* page_table_k,
    int* page_table_v,
    int* seq_lens_q,
    int* seq_lens_kv,
    int* cu_seq_lens_q,
    float softmax_scale_log2,
    int total_bh,
    int page_row_stride,
    int num_ctas,
    uint32_t* dynamic_counter,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_context_fp8(
    const void* Q,
    const void* K,
    const void* V,
    uint8_t* O_ptr,
    float* LSE_ptr,
    float* sinks,
    int* page_table_k,
    int* page_table_v,
    int* seq_lens_q,
    int* seq_lens_kv,
    int* cu_seq_lens_q,
    float softmax_scale_log2,
    float output_scale,
    int total_bh,
    int page_row_stride,
    int num_ctas,
    uint32_t* dynamic_counter,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_context_nvfp4_dequant(
    uint8_t* K_packed,
    uint8_t* V_packed,
    uint8_t* K_scales,
    uint8_t* V_scales,
    uint8_t* K_output,
    uint8_t* V_output,
    int total_groups,
    int output_page_stride,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_context_fp16_hd256(
    const void* Q,
    const void* K,
    const void* V,
    __half* O_ptr,
    int* page_table,
    int* seq_lens_q,
    int* seq_lens_kv,
    int* cu_seq_lens_q,
    float softmax_scale_log2,
    int total_bh,
    int max_pages_per_seq,
    uint32_t* dynamic_counter,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_context_fp8_hd256(
    const void* Q,
    const void* K,
    const void* V,
    uint8_t* O_ptr,
    int* page_table,
    int* seq_lens_q,
    int* seq_lens_kv,
    int* cu_seq_lens_q,
    float softmax_scale_log2,
    float output_scale,
    int total_bh,
    int max_pages_per_seq,
    uint32_t* dynamic_counter,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);

extern "C" cudaError_t cake_fmha_launch_compat_v1(
    const void* q,
    const void* k,
    const void* v,
    const void* k_scales,
    const void* v_scales,
    void* o,
    void* o_scales,
    float* lse,
    const int* page_table_k,
    const int* page_table_v,
    const int* q_indptr,
    const int* seq_lens_kv,
    const float* sinks,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int kv_layout,
    int q_dtype,
    int kv_dtype,
    int o_dtype,
    int causal,
    int window_left,
    int enable_sink,
    int return_lse,
    float q_scale,
    float k_scale,
    float v_scale,
    float o_scale,
    float sm_scale,
    float o_sf_scale,
    int o_sf_start,
    int o_sf_columns,
    long long q_s0,
    long long q_s1,
    long long k_s0,
    long long k_s1,
    long long k_s2,
    long long k_s3,
    long long v_s0,
    long long v_s1,
    long long v_s2,
    long long v_s3,
    long long ksf_s0,
    long long ksf_s1,
    long long ksf_s2,
    long long ksf_s3,
    long long vsf_s0,
    long long vsf_s1,
    long long vsf_s2,
    long long vsf_s3,
    long long table_k_s0,
    long long table_v_s0,
    long long o_s0,
    long long o_s1,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    cudaStream_t stream
);
