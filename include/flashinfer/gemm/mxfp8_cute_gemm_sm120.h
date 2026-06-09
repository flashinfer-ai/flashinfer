/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <cute/numeric/numeric_types.hpp>

namespace flashinfer::gemm::mxfp8_cute_sm120
{

class Mxfp8CuteGemmSm120RunnerInterface
{
public:
    Mxfp8CuteGemmSm120RunnerInterface() {}

    virtual ~Mxfp8CuteGemmSm120RunnerInterface() {}

    virtual void gemm_mxfp8_nt_groupwise(
        void* D, void const* A, void const* B,
        int shape_m, int shape_n, int shape_k,
        float const* scales_a, float const* scales_b,
        cudaStream_t stream, int granK = 128) = 0;

    virtual void batch_gemm_mxfp8_nt_groupwise(
        void* A, int ld_a, int stride_a,
        void* B, int ld_b, int stride_b,
        void* D, int ld_d, int stride_d,
        float* SFA, float* SFB,
        int num_groups, int shape_m,
        int shape_n, int shape_k, cudaStream_t stream, int granK = 128) = 0;

    virtual void group_gemm_mxfp8_nt_groupwise_masked(
        void* D, void const* A, void const* B, int const* masked_m,
        int num_groups, int max_m, int n, int k, cudaStream_t stream,
        float const* SFA = nullptr, float const* SFB = nullptr, int granK = 128) = 0;

    virtual void group_gemm_mxfp8_nt_groupwise_zero_padding(
        void* D, void const* A, void const* B, int32_t const* token_offset,
        int num_groups, int max_shape_m, int shape_n, int shape_k, cudaStream_t stream,
        float const* SFA = nullptr, float const* SFB = nullptr, int granK = 128) = 0;

    virtual void group_gemm_mxfp8_nt_groupwise_contiguous(
        void* D, void const* A, void const* B, int32_t const* m_indices,
        int num_groups, int m, int shape_n, int shape_k, cudaStream_t stream,
        float const* SFA = nullptr, float const* SFB = nullptr, int granK = 128,
        bool use_psum_layout = true) = 0;
};

template <typename ElementType, typename OutElementType,
          typename AccumElementType, typename BlockScaleElementType>
class Mxfp8CuteGemmSm120Runner
    : public Mxfp8CuteGemmSm120RunnerInterface
{
public:
    Mxfp8CuteGemmSm120Runner();
    ~Mxfp8CuteGemmSm120Runner();

    void gemm_mxfp8_nt_groupwise(
        void* D, void const* A, void const* B,
        int shape_m, int shape_n, int shape_k, float const* scales_a,
        float const* scales_b, cudaStream_t stream, int granK = 128) override;

    void batch_gemm_mxfp8_nt_groupwise(
        void* A, int ld_a, int stride_a,
        void* B, int ld_b, int stride_b,
        void* D, int ld_d, int stride_d,
        float* SFA, float* SFB,
        int num_groups, int shape_m,
        int shape_n, int shape_k, cudaStream_t stream, int granK = 128) override;

    void group_gemm_mxfp8_nt_groupwise_masked(
        void* D, void const* A, void const* B, int const* masked_m,
        int num_groups, int max_m, int n, int k, cudaStream_t stream,
        float const* SFA = nullptr, float const* SFB = nullptr, int granK = 128) override;

    void group_gemm_mxfp8_nt_groupwise_zero_padding(
        void* D, void const* A, void const* B, int32_t const* token_offset,
        int num_groups, int max_shape_m, int shape_n, int shape_k, cudaStream_t stream,
        float const* SFA = nullptr, float const* SFB = nullptr, int granK = 128) override;

    void group_gemm_mxfp8_nt_groupwise_contiguous(
        void* D, void const* A, void const* B, int32_t const* m_indices,
        int num_groups, int m, int shape_n, int shape_k, cudaStream_t stream,
        float const* SFA = nullptr, float const* SFB = nullptr, int granK = 128,
        bool use_psum_layout = true) override;

private:
    template <int GranK>
    void gemm_mxfp8_nt_groupwise_impl(
        void* D, void const* A, void const* B,
        int shape_m, int shape_n, int shape_k,
        float const* scales_a, float const* scales_b, cudaStream_t stream);

    template <int GranK>
    void batch_gemm_mxfp8_nt_groupwise_impl(
        void* A, int ld_a, int stride_a,
        void* B, int ld_b, int stride_b,
        void* D, int ld_d, int stride_d,
        float* SFA, float* SFB,
        int num_groups, int shape_m,
        int shape_n, int shape_k, cudaStream_t stream);

    template <int GranK>
    void group_gemm_mxfp8_nt_groupwise_masked_impl(
        void* D, void const* A, void const* B, int const* masked_m,
        int num_groups, int max_m, int n, int k, cudaStream_t stream,
        float const* SFA, float const* SFB);

    template <int GranK>
    void group_gemm_mxfp8_nt_groupwise_zero_padding_impl(
        void* D, void const* A, void const* B, int32_t const* token_offset,
        int num_groups, int max_shape_m, int shape_n, int shape_k, cudaStream_t stream,
        float const* SFA, float const* SFB);

    template <int GranK>
    void group_gemm_mxfp8_nt_groupwise_contiguous_impl(
        void* D, void const* A, void const* B, int32_t const* m_indices,
        int num_groups, int m, int shape_n, int shape_k, cudaStream_t stream,
        float const* SFA, float const* SFB, bool use_psum_layout);
};



void quantize_mxfp8_zero_padding(
    void* fp8_output,           // [total_tokens, k] FP8 output
    void* scale_output,         // [k/(granK*4), total_m_padded] int32 packed scale
    void* input,                // [total_tokens, k] bfloat16 input
    void* token_offset,         // [num_experts + 1] int32 token offsets per expert
    int64_t num_experts,
    int64_t max_token_num,      // upper bound for total tokens (for grid calculation)
    int64_t size_k,
    cudaStream_t stream,
    int granK = 128);

} // namespace flashinfer::gemm::mxfp8_cute_sm120
