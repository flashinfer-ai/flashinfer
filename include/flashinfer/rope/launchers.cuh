/*
 * Copyright (c) 2023-2026 by FlashInfer team.
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
#ifndef FLASHINFER_ROPE_KERNELS_CUDA_LAUNCHERS_CUH_
#define FLASHINFER_ROPE_KERNELS_CUDA_LAUNCHERS_CUH_

/*
 * Host Launcher Functions for RoPE (Rotary Positional Embeddings)
 * ================================================================
 *
 * This header contains the host-side launcher functions that configure and
 * launch the RoPE CUDA kernels:
 *
 * - BatchQKApplyRotary: Apply RoPE using batch indptr and position offsets
 * - BatchQKApplyRotaryInPlace: In-place version of BatchQKApplyRotary
 * - BatchQKApplyRotaryPosIds: Apply RoPE using explicit position IDs
 * - BatchQKApplyRotaryPosIdsCosSinCache: Apply RoPE with precomputed cos/sin cache
 * - BatchQKApplyLlama31Rotary: Apply Llama 3.1 style RoPE with frequency scaling
 * - BatchQKApplyLlama31RotaryPosIds: Llama 3.1 RoPE with position IDs
 * - RopeQuantize: Apply RoPE + FP8 quantization
 * - RopeQuantizeAppendPagedKVCache: RoPE + quantize + append to GQA/MHA cache
 * - RopeQuantizeAppendPagedMLACache: RoPE + quantize + append to MLA cache
 *
 * Launcher functions handle:
 * - Runtime-to-compile-time dispatch for template parameters
 * - Grid/block dimension calculation
 * - Kernel selection based on GPU occupancy (parallel vs sequential heads)
 * - PDL (Programmatic Dependent Launch) configuration for Hopper+ GPUs
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>

#include "flashinfer/page.cuh"
#include "flashinfer/rope/kernels.cuh"
#include "flashinfer/rope/types.cuh"
#include "flashinfer/utils.cuh"

namespace flashinfer {

/*!
 * \brief Apply RoPE + FP8 quantization to Q and K tensors.
 *
 * Processes rope and non-rope slices of the head dimension separately,
 * applying RoPE to the rope slice and just scaling the non-rope slice.
 *
 * \tparam DType Input data type (e.g., half, bfloat16)
 * \tparam IdType Position ID type (e.g., int32_t, int64_t)
 * \tparam QuantType Output quantized type (e.g., fp8_e4m3)
 * \param q_rope_in Input Q tensor (rope slice)
 * \param k_rope_in Input K tensor (rope slice)
 * \param q_nope_in Input Q tensor (non-rope slice)
 * \param k_nope_in Input K tensor (non-rope slice)
 * \param q_rope_out Output Q tensor (rope slice, quantized)
 * \param k_rope_out Output K tensor (rope slice, quantized)
 * \param q_nope_out Output Q tensor (non-rope slice, quantized)
 * \param k_nope_out Output K tensor (non-rope slice, quantized)
 * \param cos_sin_cache Precomputed cos/sin cache [max_seq_len, rope_dim]
 * \param pos_ids Position IDs for each token
 * \param nnz Total number of tokens
 * \param num_qo_heads Number of query/output heads
 * \param num_kv_heads Number of key/value heads
 * \param rope_dim Dimension with rotary embeddings
 * \param no_rope_dim Dimension without rotary embeddings
 * \param quant_scale_q Quantization scale for Q
 * \param quant_scale_kv Quantization scale for K/V
 * \param interleave Whether to use interleaved RoPE layout
 * \param enable_pdl Enable Programmatic Dependent Launch (Hopper+)
 * \param stream CUDA stream for execution
 * \return cudaError_t CUDA error status
 */
template <typename DType, typename IdType, typename QuantType>
cudaError_t RopeQuantize(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, QuantType* q_rope_out,
    QuantType* k_rope_out, QuantType* q_nope_out, QuantType* k_nope_out, float* cos_sin_cache,
    IdType* pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rope_dim,
    uint32_t no_rope_dim, size_t q_rope_in_stride_n, size_t q_rope_in_stride_h,
    size_t q_nope_in_stride_n, size_t q_nope_in_stride_h, size_t q_rope_out_stride_n,
    size_t q_rope_out_stride_h, size_t q_nope_out_stride_n, size_t q_nope_out_stride_h,
    size_t k_rope_in_stride, size_t k_rope_in_stride_h, size_t k_nope_in_stride,
    size_t k_nope_in_stride_h, size_t k_rope_out_stride, size_t k_rope_out_stride_h,
    size_t k_nope_out_stride, size_t k_nope_out_stride_h, float quant_scale_q, float quant_scale_kv,
    bool interleave, bool enable_pdl = false, cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    constexpr uint32_t vec_size = 32 / sizeof(DType);
    uint32_t bdx = (rope_dim + vec_size - 1) / vec_size;
    bdx = std::max(1u, bdx);
    uint32_t num_threads = std::max(128U, bdx);
    uint32_t bdy = std::max(1u, num_threads / bdx);
    uint32_t nblks_x = (nnz + bdy - 1) / bdy;
    uint32_t rope_chunk_size = rope_dim;
    uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
    uint32_t no_rope_chunks = (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;
    uint32_t total_blocks_y = num_qo_heads * rope_chunks + num_kv_heads * rope_chunks +
                              num_kv_heads * no_rope_chunks + num_qo_heads * no_rope_chunks;
    void* args[] = {(void*)&q_rope_in,
                    (void*)&k_rope_in,
                    (void*)&q_nope_in,
                    (void*)&k_nope_in,
                    (void*)&q_rope_out,
                    (void*)&k_rope_out,
                    (void*)&q_nope_out,
                    (void*)&k_nope_out,
                    (void*)&cos_sin_cache,
                    (void*)&pos_ids,
                    (void*)&nnz,
                    (void*)&num_qo_heads,
                    (void*)&num_kv_heads,
                    (void*)&rope_dim,
                    (void*)&no_rope_dim,
                    (void*)&q_rope_in_stride_n,
                    (void*)&q_rope_in_stride_h,
                    (void*)&q_nope_in_stride_n,
                    (void*)&q_nope_in_stride_h,
                    (void*)&q_rope_out_stride_n,
                    (void*)&q_rope_out_stride_h,
                    (void*)&q_nope_out_stride_n,
                    (void*)&q_nope_out_stride_h,
                    (void*)&k_rope_in_stride,
                    (void*)&k_rope_in_stride_h,
                    (void*)&k_nope_in_stride,
                    (void*)&k_nope_in_stride_h,
                    (void*)&k_rope_out_stride,
                    (void*)&k_rope_out_stride_h,
                    (void*)&k_nope_out_stride,
                    (void*)&k_nope_out_stride_h,
                    (void*)&quant_scale_q,
                    (void*)&quant_scale_kv};
    auto kernel = RopeQuantizeKernel<INTERLEAVE, vec_size, 1, DType, IdType, QuantType>;
    dim3 nblks(nblks_x, total_blocks_y);
    dim3 nthrs(bdx, bdy);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
    cudaLaunchConfig_t config;
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.stream = stream;
    config.dynamicSmemBytes = 0;
    config.attrs = attribute;
    config.numAttrs = 1;

    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
        &config, kernel, q_rope_in, k_rope_in, q_nope_in, k_nope_in, q_rope_out, k_rope_out,
        q_nope_out, k_nope_out, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rope_dim,
        no_rope_dim, q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n, q_nope_in_stride_h,
        q_rope_out_stride_n, q_rope_out_stride_h, q_nope_out_stride_n, q_nope_out_stride_h,
        k_rope_in_stride, k_rope_in_stride_h, k_nope_in_stride, k_nope_in_stride_h,
        k_rope_out_stride, k_rope_out_stride_h, k_nope_out_stride, k_nope_out_stride_h,
        quant_scale_q, quant_scale_kv));
  });

  return cudaSuccess;
}

/*!
 * \brief Apply RoPE + quantize + append to GQA/MHA paged KV cache.
 *
 * \tparam DType Input data type
 * \tparam RoPEIdType Position ID type for RoPE
 * \tparam PagedKVIdType Index type for paged KV cache
 * \tparam QuantType Output quantized type
 * \param paged_kv Paged KV cache structure
 * \param batch_indices Batch index for each token
 * \param positions Position within sequence for each token
 * \param enable_pdl Enable Programmatic Dependent Launch
 * \param stream CUDA stream
 * \return cudaError_t CUDA error status
 */
template <typename DType, typename RoPEIdType, typename PagedKVIdType, typename QuantType>
cudaError_t RopeQuantizeAppendPagedKVCache(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, DType* v_in,
    QuantType* q_rope_out, QuantType* q_nope_out, paged_kv_t<QuantType, PagedKVIdType> paged_kv,
    PagedKVIdType* batch_indices, PagedKVIdType* positions, float* cos_sin_cache,
    RoPEIdType* pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rope_dim, uint32_t no_rope_dim, size_t q_rope_in_stride_n, size_t q_rope_in_stride_h,
    size_t q_nope_in_stride_n, size_t q_nope_in_stride_h, size_t q_rope_out_stride_n,
    size_t q_rope_out_stride_h, size_t q_nope_out_stride_n, size_t q_nope_out_stride_h,
    size_t k_rope_in_stride, size_t k_rope_in_stride_h, size_t k_nope_in_stride,
    size_t k_nope_in_stride_h, size_t v_in_stride, size_t v_in_stride_h, float quant_scale_q,
    float quant_scale_kv, bool interleave, bool enable_pdl = false, cudaStream_t stream = nullptr) {
  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    constexpr uint32_t vec_size = 32 / sizeof(DType);
    uint32_t bdx = (rope_dim + vec_size - 1) / vec_size;
    bdx = std::max(1u, bdx);
    uint32_t num_threads = std::max(128U, bdx);
    uint32_t bdy = std::max(1u, num_threads / bdx);
    uint32_t nblks_x = (nnz + bdy - 1) / bdy;
    uint32_t rope_chunks = 1;
    uint32_t no_rope_chunks = (no_rope_dim + rope_dim - 1) / rope_dim;

    uint32_t total_blocks_y = num_qo_heads * rope_chunks + num_kv_heads * rope_chunks +
                              num_kv_heads * no_rope_chunks + num_kv_heads +
                              num_qo_heads * no_rope_chunks;

    dim3 nblks(nblks_x, total_blocks_y);
    dim3 nthrs(bdx, bdy);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
    cudaLaunchConfig_t config;
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.stream = stream;
    config.dynamicSmemBytes = 0;
    config.attrs = attribute;
    config.numAttrs = 1;

    auto kernel = RopeQuantizeAppendPagedKVCacheKernel<INTERLEAVE, vec_size, /*bdx=*/1, DType,
                                                       RoPEIdType, PagedKVIdType, QuantType,
                                                       paged_kv_t<QuantType, PagedKVIdType>>;
    RopeQuantizeAppendPagedKVCacheParams params;
    params.nnz = nnz;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.rope_dim = rope_dim;
    params.no_rope_dim = no_rope_dim;
    params.q_rope_in_stride_n = q_rope_in_stride_n;
    params.q_rope_in_stride_h = q_rope_in_stride_h;
    params.q_nope_in_stride_n = q_nope_in_stride_n;
    params.q_nope_in_stride_h = q_nope_in_stride_h;
    params.q_rope_out_stride_n = q_rope_out_stride_n;
    params.q_rope_out_stride_h = q_rope_out_stride_h;
    params.q_nope_out_stride_n = q_nope_out_stride_n;
    params.q_nope_out_stride_h = q_nope_out_stride_h;
    params.k_rope_in_stride = k_rope_in_stride;
    params.k_rope_in_stride_h = k_rope_in_stride_h;
    params.k_nope_in_stride = k_nope_in_stride;
    params.k_nope_in_stride_h = k_nope_in_stride_h;
    params.v_in_stride = v_in_stride;
    params.v_in_stride_h = v_in_stride_h;
    params.quant_scale_q = quant_scale_q;
    params.quant_scale_kv = quant_scale_kv;

    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
        &config, kernel, q_rope_in, k_rope_in, q_nope_in, k_nope_in, v_in, q_rope_out, q_nope_out,
        paged_kv, batch_indices, positions, cos_sin_cache, pos_ids, params));
  });

  return cudaSuccess;
}

/*!
 * \brief Apply RoPE + quantize + append to MLA paged cache.
 *
 * MLA (Multi-head Latent Attention) uses a different cache structure with
 * separate k_pe (key positional encoding) and c_kv (compressed key-value) slots.
 *
 * \tparam DType Input data type
 * \tparam RoPEIdType Position ID type
 * \tparam PagedKVIdType Page index type
 * \tparam QuantType Quantized output type
 * \param paged_kv_mla MLA paged cache structure
 * \param enable_pdl Enable PDL for Hopper+
 * \param stream CUDA stream
 * \return cudaError_t CUDA error status
 */
template <typename DType, typename RoPEIdType, typename PagedKVIdType, typename QuantType>
cudaError_t RopeQuantizeAppendPagedMLACache(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, QuantType* q_rope_out,
    QuantType* q_nope_out, paged_kv_mla_t<QuantType, PagedKVIdType> paged_kv_mla,
    PagedKVIdType* batch_indices, PagedKVIdType* positions, float* cos_sin_cache,
    RoPEIdType* pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t rope_dim,
    uint32_t no_rope_dim, size_t q_rope_in_stride_n, size_t q_rope_in_stride_h,
    size_t q_nope_in_stride_n, size_t q_nope_in_stride_h, size_t q_rope_out_stride_n,
    size_t q_rope_out_stride_h, size_t q_nope_out_stride_n, size_t q_nope_out_stride_h,
    size_t k_rope_in_stride, size_t k_nope_in_stride, float quant_scale_q, float quant_scale_kv,
    bool interleave, bool enable_pdl = false, cudaStream_t stream = nullptr) {
  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    constexpr uint32_t vec_size = 32 / sizeof(DType);
    uint32_t bdx = (rope_dim + vec_size - 1) / vec_size;
    bdx = std::max(1u, bdx);
    uint32_t num_threads = std::max(128U, bdx);
    uint32_t bdy = std::max(1u, num_threads / bdx);
    uint32_t nblks_x = (nnz + bdy - 1) / bdy;
    uint32_t rope_chunks = 1;
    uint32_t no_rope_chunks = (no_rope_dim + rope_dim - 1) / rope_dim;
    constexpr uint32_t num_kv_heads = 1;
    uint32_t total_blocks_y = num_qo_heads * rope_chunks + num_kv_heads * rope_chunks +
                              num_kv_heads * no_rope_chunks + num_qo_heads * no_rope_chunks;

    dim3 nblks(nblks_x, total_blocks_y);
    dim3 nthrs(bdx, bdy);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
    cudaLaunchConfig_t config;
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.stream = stream;
    config.dynamicSmemBytes = 0;
    config.attrs = attribute;
    config.numAttrs = 1;

    auto kernel = RopeQuantizeAppendPagedKVCacheKernel<INTERLEAVE, vec_size, /*bdx=*/1, DType,
                                                       RoPEIdType, PagedKVIdType, QuantType,
                                                       paged_kv_mla_t<QuantType, PagedKVIdType>>;
    DType* v_in_nullptr = nullptr;
    uint32_t num_kv_heads_1 = 1;
    size_t k_rope_in_stride_h_dup = k_rope_in_stride;
    size_t k_nope_in_stride_h_dup = k_nope_in_stride;
    RopeQuantizeAppendPagedKVCacheParams params;
    params.nnz = nnz;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = 1u;
    params.rope_dim = rope_dim;
    params.no_rope_dim = no_rope_dim;
    params.q_rope_in_stride_n = q_rope_in_stride_n;
    params.q_rope_in_stride_h = q_rope_in_stride_h;
    params.q_nope_in_stride_n = q_nope_in_stride_n;
    params.q_nope_in_stride_h = q_nope_in_stride_h;
    params.q_rope_out_stride_n = q_rope_out_stride_n;
    params.q_rope_out_stride_h = q_rope_out_stride_h;
    params.q_nope_out_stride_n = q_nope_out_stride_n;
    params.q_nope_out_stride_h = q_nope_out_stride_h;
    params.k_rope_in_stride = k_rope_in_stride;
    params.k_rope_in_stride_h = k_rope_in_stride_h_dup;
    params.k_nope_in_stride = k_nope_in_stride;
    params.k_nope_in_stride_h = k_nope_in_stride_h_dup;
    params.v_in_stride = 0;
    params.v_in_stride_h = 0;
    params.quant_scale_q = quant_scale_q;
    params.quant_scale_kv = quant_scale_kv;

    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel,
                                            // inputs
                                            q_rope_in, k_rope_in, q_nope_in, k_nope_in,
                                            v_in_nullptr,
                                            // q outputs
                                            q_rope_out, q_nope_out,
                                            // cache + indices
                                            paged_kv_mla, batch_indices, positions,
                                            // rope tables
                                            cos_sin_cache, pos_ids,
                                            // params
                                            params));
  });

  return cudaSuccess;
}

/*!
 * \brief Apply RoPE with precomputed cos/sin cache and position IDs.
 *
 * Automatically selects between sequential-heads and head-parallel kernels
 * based on GPU occupancy for optimal performance.
 *
 * \tparam DType Data type (half, bfloat16)
 * \tparam IdType Position ID type
 * \param q Input Q tensor [nnz, num_qo_heads, head_dim]
 * \param k Input K tensor [nnz, num_kv_heads, head_dim]
 * \param q_rope Output Q tensor with RoPE applied
 * \param k_rope Output K tensor with RoPE applied
 * \param cos_sin_cache Precomputed cache [max_seq_len, rotary_dim]
 * \param pos_ids Position ID for each token
 * \param nnz Total number of tokens
 * \param interleave Use interleaved RoPE layout
 * \param stream CUDA stream
 * \return cudaError_t CUDA error status
 */
template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryPosIdsCosSinCache(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* cos_sin_cache, IdType* pos_ids,
    uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim,
    uint32_t head_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    bool interleave, cudaStream_t stream = nullptr) {
  if (head_dim < rotary_dim) {
    std::ostringstream err_msg;
    err_msg << "head_dim (" << head_dim << ") must be >= rotary_dim (" << rotary_dim << ")";
    FLASHINFER_ERROR(err_msg.str());
  }

  // Use optimized kernel for common head dimensions
  if (head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512) {
    int dev_id = 0;
    int num_sms = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

    DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
        constexpr uint32_t bdx = HEAD_DIM / vec_size;
        uint32_t num_threads = std::max(128U, bdx);
        uint32_t bdy = num_threads / bdx;
        uint32_t nblks_x = (nnz + bdy - 1) / bdy;
        void* args[] = {(void*)&q,
                        (void*)&k,
                        (void*)&q_rope,
                        (void*)&k_rope,
                        (void*)&cos_sin_cache,
                        (void*)&pos_ids,
                        (void*)&nnz,
                        (void*)&num_qo_heads,
                        (void*)&num_kv_heads,
                        (void*)&rotary_dim,
                        (void*)&q_stride_n,
                        (void*)&q_stride_h,
                        (void*)&k_stride_n,
                        (void*)&k_stride_h,
                        (void*)&q_rope_stride_n,
                        (void*)&q_rope_stride_h,
                        (void*)&k_rope_stride_n,
                        (void*)&k_rope_stride_h};
        auto kernel_0 = BatchQKApplyRotaryPosIdsCosSinCacheKernel<INTERLEAVE, HEAD_DIM, vec_size,
                                                                  bdx, DType, IdType>;

        int num_blocks_per_sm_0 = 0;
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm_0, kernel_0, num_threads, /*smem_size=*/0));
        uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

        if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
          // Large workload: use sequential-heads kernel
          dim3 nblks(nblks_x);
          dim3 nthrs(bdx, bdy);
          FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_0, nblks, nthrs, args, 0, stream));
        } else {
          // Small workload: use head-parallel kernel
          dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
          dim3 nthrs(bdx, bdy);
          auto kernel_1 = BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<
              INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
          FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_1, nblks, nthrs, args, 0, stream));
        }
      });
    });
    return cudaSuccess;
  }

  // Fall back to RopeQuantize kernel for non-standard head dimensions
  const uint32_t rope_dim = rotary_dim;
  const uint32_t no_rope_dim = head_dim - rotary_dim;

  DType* q_rope_in = q;
  DType* k_rope_in = k;
  DType* q_nope_in = q + rotary_dim;
  DType* k_nope_in = k + rotary_dim;
  DType* q_rope_out = q_rope;
  DType* k_rope_out = k_rope;
  DType* q_nope_out = q_rope + rotary_dim;
  DType* k_nope_out = k_rope + rotary_dim;

  return RopeQuantize<DType, IdType, DType>(
      q_rope_in, k_rope_in, q_nope_in, k_nope_in, q_rope_out, k_rope_out, q_nope_out, k_nope_out,
      cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rope_dim, no_rope_dim, q_stride_n,
      q_stride_h, q_stride_n, q_stride_h, q_rope_stride_n, q_rope_stride_h, q_rope_stride_n,
      q_rope_stride_h, k_stride_n, k_stride_h, k_stride_n, k_stride_h, k_rope_stride_n,
      k_rope_stride_h, k_rope_stride_n, k_rope_stride_h, /*quant_scale_q=*/1.0f,
      /*quant_scale_kv=*/1.0f, interleave, /*enable_pdl=*/false, stream);
}

/*!
 * \brief Apply RoPE with explicit position IDs.
 *
 * Computes frequency on-the-fly from rope_scale and rope_theta parameters.
 * Automatically selects kernel based on workload size.
 *
 * \tparam DType Data type
 * \tparam IdType Position ID type
 * \param pos_ids Position ID array [nnz]
 * \param rope_scale RoPE scaling factor
 * \param rope_theta RoPE theta parameter (default: 10000.0)
 * \param stream CUDA stream
 * \return cudaError_t CUDA error status
 */
template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryPosIds(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim,
    size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    bool interleave, float rope_scale, float rope_theta, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = 0.f;
  float smooth_b = 0.f;
  int dev_id = 0;
  int num_sms = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;

      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&pos_ids,
                      (void*)&nnz,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      auto kernel_0 =
          BatchQKApplyRotaryPosIdsKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;

      int num_blocks_per_sm_0 = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm_0, kernel_0, num_threads, /*smem_size=*/0));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;
      if (nblks_x >= num_ctas_0) {
        dim3 nblks(nblks_x);
        dim3 nthrs(bdx, bdy);

        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_0, nblks, nthrs, args, 0, stream));
      } else {
        dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel_1 = BatchQKApplyRotaryPosIdsHeadParallelismKernel<INTERLEAVE, HEAD_DIM,
                                                                      vec_size, bdx, DType, IdType>;

        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_1, nblks, nthrs, args, 0, stream));
      }
    });
  });

  return cudaSuccess;
}

/*!
 * \brief Apply RoPE using batch indptr and position offsets.
 *
 * For ragged batched tensors where sequences have different lengths.
 * Each sequence's position starts from offset[batch_idx].
 *
 * \tparam DType Data type
 * \tparam IdType Index type
 * \param indptr Cumulative sequence lengths [batch_size + 1]
 * \param offsets Starting position for each sequence [batch_size]
 * \param batch_size Number of sequences in batch
 * \param stream CUDA stream
 * \return cudaError_t CUDA error status
 */
template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotary(DType* q, DType* k, DType* q_rope, DType* k_rope,
                               IdType* __restrict__ indptr, IdType* __restrict__ offsets,
                               uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                               uint32_t rotary_dim, uint32_t head_dim, size_t q_stride_n,
                               size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
                               size_t q_rope_stride_n, size_t q_rope_stride_h,
                               size_t k_rope_stride_n, size_t k_rope_stride_h, bool interleave,
                               float rope_scale, float rope_theta, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = 0.f;
  float smooth_b = 0.f;

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
      dim3 nthrs(bdx, bdy);
      auto kernel = BatchQKApplyRotaryKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&indptr,
                      (void*)&offsets,
                      (void*)&batch_size,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    });
  });

  return cudaSuccess;
}

/*!
 * \brief Apply RoPE in-place using batch indptr and position offsets.
 *
 * Same as BatchQKApplyRotary but modifies input tensors directly.
 *
 * \tparam DType Data type
 * \tparam IdType Index type
 * \return cudaError_t CUDA error status
 */
template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryInPlace(DType* __restrict__ q, DType* __restrict__ k,
                                      IdType* __restrict__ indptr, IdType* __restrict__ offsets,
                                      uint32_t batch_size, uint32_t num_qo_heads,
                                      uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim,
                                      size_t q_stride_n, size_t q_stride_h, size_t k_stride_n,
                                      size_t k_stride_h, bool interleave, float rope_scale,
                                      float rope_theta, cudaStream_t stream = nullptr) {
  return BatchQKApplyRotary<DType, IdType>(
      q, k, q, k, indptr, offsets, batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim,
      q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_stride_n, q_stride_h, k_stride_n,
      k_stride_h, interleave, rope_scale, rope_theta, stream);
}

/*!
 * \brief Apply Llama 3.1 style RoPE with adaptive frequency scaling.
 *
 * Llama 3.1 introduced frequency scaling that varies based on frequency value:
 * - Low frequencies: Scaled by rope_scale
 * - High frequencies: Not scaled (pass-through)
 * - Mid frequencies: Smoothly interpolated
 *
 * \tparam DType Data type
 * \tparam IdType Index type
 * \param low_freq_factor Lower frequency bound for scaling
 * \param high_freq_factor Upper frequency bound for scaling
 * \param old_context_length Original context length for smooth interpolation
 * \param stream CUDA stream
 * \return cudaError_t CUDA error status
 */
template <typename DType, typename IdType>
cudaError_t BatchQKApplyLlama31Rotary(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, uint32_t head_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n,
    size_t k_stride_h, size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n,
    size_t k_rope_stride_h, bool interleave, float rope_scale, float rope_theta,
    float low_freq_factor, float high_freq_factor, float old_context_length,
    cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = old_context_length / (2 * M_PI * high_freq_factor - 2 * M_PI * low_freq_factor);
  float smooth_b = -1.0f / (high_freq_factor / low_freq_factor - 1.0f);

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
      dim3 nthrs(bdx, bdy);
      auto kernel = BatchQKApplyRotaryKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&indptr,
                      (void*)&offsets,
                      (void*)&batch_size,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    });
  });

  return cudaSuccess;
}

/*!
 * \brief Apply Llama 3.1 style RoPE with explicit position IDs.
 *
 * Combines Llama 3.1 frequency scaling with explicit position ID support.
 *
 * \tparam DType Data type
 * \tparam IdType Position ID type
 * \param pos_ids Position ID for each token
 * \param low_freq_factor Lower frequency bound
 * \param high_freq_factor Upper frequency bound
 * \param old_context_length Original context length
 * \param stream CUDA stream
 * \return cudaError_t CUDA error status
 */
template <typename DType, typename IdType>
cudaError_t BatchQKApplyLlama31RotaryPosIds(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim,
    size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    bool interleave, float rope_scale, float rope_theta, float low_freq_factor,
    float high_freq_factor, float old_context_length, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = old_context_length / (2 * M_PI * high_freq_factor - 2 * M_PI * low_freq_factor);
  float smooth_b = -1.0f / (high_freq_factor / low_freq_factor - 1.0f);

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      dim3 nblks((nnz + bdy - 1) / bdy);
      dim3 nthrs(bdx, bdy);
      auto kernel =
          BatchQKApplyRotaryPosIdsKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&pos_ids,
                      (void*)&nnz,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    });
  });

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ROPE_KERNELS_CUDA_LAUNCHERS_CUH_
