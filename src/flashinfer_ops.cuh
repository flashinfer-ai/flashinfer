/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <optional>

#include "flashinfer/allocator.h"
#include "flashinfer/attention/mask.cuh"
#include "flashinfer/attention/scheduler.cuh"
#include "flashinfer/exception.h"
#include "flashinfer/layout.cuh"
#include "utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, cudaStream_t stream);

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename AttentionVariant, typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatchedMLA(Params params, typename Params::DTypeO* tmp_v,
                                                     float* tmp_s, cudaStream_t stream);

class BatchDecodeHandler {
 public:
  template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
            typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
  cudaError_t PlanDispatched(void* float_buffer, size_t float_workspace_size_in_bytes,
                             void* int_buffer, size_t int_workspace_size_in_bytes, IdType* indptr_h,
                             IdType* last_page_len_h, uint32_t batch_size, uint32_t num_qo_heads,
                             uint32_t page_size) {
    int_buffer_ = int_buffer;
    float_buffer_ = float_buffer;
    using Params = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
    using AttentionVariant =
        DefaultAttention</*use_custom_mask=*/false, /*use_sliding_window=*/false,
                         /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;

    auto work_estimation_func =
        BatchDecodeWithPagedKVCacheWorkEstimationDispatched<GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE,
                                                            AttentionVariant, Params>;
    return DecodePlan<HEAD_DIM, POS_ENCODING_MODE, AttentionVariant, Params>(
        float_buffer, float_workspace_size_in_bytes, int_buffer, page_locked_buffer_,
        int_workspace_size_in_bytes, plan_info_, indptr_h, batch_size, num_qo_heads, page_size,
        cuda_graph_enabled_, stream_, work_estimation_func);
  }

  template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename DTypeQ, typename DTypeKV,
            typename DTypeO, typename IdType>
  cudaError_t PlanDispatchedMLA(void* float_buffer, size_t float_workspace_size_in_bytes,
                                void* int_buffer, size_t int_workspace_size_in_bytes,
                                IdType* indptr_h, IdType* last_page_len_h, uint32_t batch_size,
                                uint32_t num_qo_heads, uint32_t page_size) {
    int_buffer_ = int_buffer;
    float_buffer_ = float_buffer;
    using Params = BatchDecodeParamsMLA<DTypeQ, DTypeKV, DTypeO, IdType>;
    using AttentionVariant =
        DefaultAttention</*use_custom_mask=*/false, /*use_sliding_window=*/false,
                         /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;

    auto work_estimation_func =
        BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMLA<HEAD_DIM_CKV, HEAD_DIM_KPE,
                                                               AttentionVariant, Params>;
    return DecodePlan<HEAD_DIM_CKV, flashinfer::PosEncodingMode::kRoPELlama, AttentionVariant,
                      Params>(float_buffer, float_workspace_size_in_bytes, int_buffer,
                              page_locked_buffer_, int_workspace_size_in_bytes, plan_info_,
                              indptr_h, batch_size, num_qo_heads, page_size, cuda_graph_enabled_,
                              stream_, work_estimation_func);
  }

  void UpdatePageLockedBufferSize(size_t int_workspace_size_in_bytes) {
    cudaFreeHost(page_locked_buffer_);
    cudaMallocHost(&page_locked_buffer_, int_workspace_size_in_bytes);
  }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  /*!
   * \brief Constructor of BatchDecodeHandler
   * \param enable_cuda_graph A boolean indicates whether to enable CUDA graph
   * \param batch_size If enable_cuda_graph is true, we must specify a fixed batch_size
   */
  BatchDecodeHandler(bool enable_cuda_graph = false, uint32_t batch_size = 0)
      : cuda_graph_enabled_(enable_cuda_graph), stream_(nullptr) {
    cudaMallocHost(&page_locked_buffer_, 8 * 1024 * 1024);
  }
  ~BatchDecodeHandler() { cudaFreeHost(page_locked_buffer_); }

  bool IsCUDAGraphEnabled() const { return cuda_graph_enabled_; }

  DecodePlanInfo GetPlanInfo() const { return plan_info_; }

  template <typename IdType>
  IdType* GetRequestIndices() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.request_indices_offset);
  }

  template <typename IdType>
  IdType* GetKVTileIndices() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.kv_tile_indices_offset);
  }

  template <typename IdType>
  IdType* GetOIndptr() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.o_indptr_offset);
  }

  template <typename IdType>
  IdType* GetKVChunkSizePtr() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.kv_chunk_size_ptr_offset);
  }

  template <typename DTypeO>
  DTypeO* GetTmpV() {
    if (plan_info_.split_kv) {
      return GetPtrFromBaseOffset<DTypeO>(float_buffer_, plan_info_.v_offset);
    }
    return nullptr;
  }

  float* GetTmpS() {
    if (plan_info_.split_kv) {
      return GetPtrFromBaseOffset<float>(float_buffer_, plan_info_.s_offset);
    }
    return nullptr;
  }

  bool* GetBlockValidMask() {
    if (plan_info_.split_kv && plan_info_.enable_cuda_graph) {
      return GetPtrFromBaseOffset<bool>(int_buffer_, plan_info_.block_valid_mask_offset);
    }
    return nullptr;
  }

 protected:
  void* page_locked_buffer_;
  void* int_buffer_;
  void* float_buffer_;
  DecodePlanInfo plan_info_;
  bool cuda_graph_enabled_;
  cudaStream_t stream_;
};

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                    float* tmp_s, cudaStream_t stream);

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, cudaStream_t stream);

class BatchPrefillHandler {
 public:
  void UpdatePageLockedBufferSize(size_t int_workspace_size_in_bytes) {
    cudaFreeHost(page_locked_buffer_);
    cudaMallocHost(&page_locked_buffer_, int_workspace_size_in_bytes);
  }

  template <typename DTypeO, typename IdType>
  cudaError_t Plan(void* float_buffer, size_t float_workspace_size_in_bytes, void* int_buffer,
                   size_t int_workspace_size_in_bytes, IdType* qo_indptr_h, IdType* kv_indptr_h,
                   uint32_t total_num_rows, uint32_t batch_size, uint32_t num_qo_heads,
                   uint32_t num_kv_heads, uint32_t head_dim, uint32_t page_size) {
    int_buffer_ = int_buffer;
    float_buffer_ = float_buffer;
    return PrefillPlan<IdType>(float_buffer, float_workspace_size_in_bytes, int_buffer,
                               page_locked_buffer_, int_workspace_size_in_bytes, plan_info_,
                               qo_indptr_h, kv_indptr_h, total_num_rows, batch_size, num_qo_heads,
                               num_kv_heads, head_dim, page_size, enable_cuda_graph_,
                               sizeof(DTypeO), stream_);
  }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  bool IsCUDAGraphEnabled() const { return enable_cuda_graph_; }

  BatchPrefillHandler(bool enable_cuda_graph = false)
      : enable_cuda_graph_(enable_cuda_graph), stream_(nullptr) {
    cudaMallocHost(&page_locked_buffer_, 8 * 1024 * 1024);
  }
  ~BatchPrefillHandler() { cudaFreeHost(page_locked_buffer_); }

  PrefillPlanInfo GetPlanInfo() const { return plan_info_; }

  template <typename IdType>
  IdType* GetRequestIndices() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.request_indices_offset);
  }

  template <typename IdType>
  IdType* GetQOTileIndices() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.qo_tile_indices_offset);
  }

  template <typename IdType>
  IdType* GetKVTileIndices() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.kv_tile_indices_offset);
  }

  template <typename IdType>
  IdType* GetOIndptr() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.o_indptr_offset);
  }

  template <typename IdType>
  IdType* GetKVChunkSizePtr() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.kv_chunk_size_ptr_offset);
  }

  template <typename IdType>
  IdType* GetMergeIndptr() {
    if (plan_info_.split_kv) {
      return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.merge_indptr_offset);
    }
    return nullptr;
  }

  template <typename DTypeO>
  DTypeO* GetTmpV() {
    if (plan_info_.split_kv) {
      return GetPtrFromBaseOffset<DTypeO>(float_buffer_, plan_info_.v_offset);
    }
    return nullptr;
  }

  float* GetTmpS() {
    if (plan_info_.split_kv) {
      return GetPtrFromBaseOffset<float>(float_buffer_, plan_info_.s_offset);
    }
    return nullptr;
  }

  uint32_t* GetTotalNumRows() {
    if (plan_info_.enable_cuda_graph) {
      return GetPtrFromBaseOffset<uint32_t>(int_buffer_, plan_info_.total_num_rows_offset);
    }
    return nullptr;
  }

  bool* GetBlockValidMask() {
    if (plan_info_.split_kv && plan_info_.enable_cuda_graph) {
      return GetPtrFromBaseOffset<bool>(int_buffer_, plan_info_.block_valid_mask_offset);
    }
    return nullptr;
  }

 protected:
  void* page_locked_buffer_;
  void* int_buffer_;
  void* float_buffer_;
  PrefillPlanInfo plan_info_;
  bool enable_cuda_graph_;
  cudaStream_t stream_;
};

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION,
          MaskMode MASK_MODE, typename AttentionVariant, typename Params>
cudaError_t SinglePrefillWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                               cudaStream_t stream);

template <typename DTypeIn, typename DTypeO>
cudaError_t SinglePrefillWithKVCacheCustomMask(
    DTypeIn* q, DTypeIn* k, DTypeIn* v, uint8_t* custom_mask, DTypeO* o, DTypeO* tmp, float* lse,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    uint32_t head_dim, QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone, bool use_fp16_qk_reduction = false,
    std::optional<float> maybe_sm_scale = std::nullopt, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, kv_len, num_qo_heads, num_kv_heads, head_dim);
  DISPATCH_use_fp16_qk_reduction(
      use_fp16_qk_reduction, USE_FP16_QK_REDUCTION,
      {DISPATCH_head_dim(
          head_dim, HEAD_DIM, {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
            using Params = SinglePrefillParams<DTypeIn, DTypeIn, DTypeO>;
            using AttentionVariant = DefaultAttention<
                /*use_custom_mask=*/true, /*use_sliding_window=*/false,
                /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
            Params params(q, k, v, custom_mask, o, lse,
                          /*alibi_slopes=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len,
                          qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h, head_dim,
                          /*window_left=*/-1,
                          /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta);
            return SinglePrefillWithKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE,
                                                      USE_FP16_QK_REDUCTION, MaskMode::kCustom,
                                                      AttentionVariant>(params, tmp, stream);
          })})});
  return cudaSuccess;
}

/*!
 * \brief FlashAttention prefill CUDA function for a single request.
 * \tparam DTypeIn The data type of input
 * \tparam DTypeO The data type of output
 * \param q The query tensor.
 * \param k The key tensor.
 * \param v The value tensor.
 * \param o The output tensor.
 * \param tmp The temporary storage (only used for cooperative kernel).
 * \param lse The logsumexp values.
 * \param num_qo_heads The number of query and output heads.
 * \param num_kv_heads The number of key and value heads.
 * \param qo_len The length of query and output.
 * \param kv_len The length of key and value.
 * \param head_dim The dimension of each head.
 * \param causal Whether to use causal attention.
 * \param kv_layout The layout of input and output.
 * \param pos_encoding_mode The positional encoding mode.
 * \param use_fp16_qk_reduction Whether to allow accumulating q*k^T with fp16.
 * \param rope_scale The scaling factor used in RoPE interpolation.
 * \param rope_theta The theta used in RoPE.
 * \param stream The cuda stream to execute the kernel on.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t SinglePrefillWithKVCache(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, DTypeO* tmp,
                                     float* lse, uint32_t num_qo_heads, uint32_t num_kv_heads,
                                     uint32_t qo_len, uint32_t kv_len, uint32_t head_dim,
                                     bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
                                     PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                     bool use_fp16_qk_reduction = false,
                                     std::optional<float> maybe_sm_scale = std::nullopt,
                                     float rope_scale = 1.f, float rope_theta = 1e4,
                                     cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, kv_len, num_qo_heads, num_kv_heads, head_dim);
  DISPATCH_use_fp16_qk_reduction(
      use_fp16_qk_reduction, USE_FP16_QK_REDUCTION,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_head_dim(
              head_dim, HEAD_DIM,
              {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
                using Params = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;
                using AttentionVariant = DefaultAttention<
                    /*use_custom_mask=*/(MASK_MODE == MaskMode::kCustom),
                    /*use_sliding_window=*/false,
                    /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
                Params params(q, k, v, /*custom_mask=*/nullptr, o, lse,
                              /*alibi_slopes=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len,
                              qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h, head_dim,
                              /*window_left=*/-1,
                              /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta);
                return SinglePrefillWithKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE,
                                                          USE_FP16_QK_REDUCTION, MASK_MODE,
                                                          AttentionVariant, Params>(params, tmp,
                                                                                    stream);
              })})})});
  return cudaSuccess;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeQ* q, IdType* qo_indptr, DTypeKV* k, DTypeKV* v,
    IdType* kv_indptr, IdType* q_rope_offset, IdType* k_rope_offset, DTypeO* o, float* lse,
    const uint32_t batch_size, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
    const uint32_t head_dim, bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone, bool use_fp16_qk_reduction = false,
    std::optional<float> maybe_sm_scale = std::nullopt, const float rope_scale = 1.f,
    const float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, 0, num_qo_heads, num_kv_heads, head_dim);
  auto plan_info = handler->GetPlanInfo();
  DISPATCH_head_dim(
      head_dim, HEAD_DIM,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_pos_encoding_mode(
              pos_encoding_mode, POS_ENCODING_MODE,
              {DISPATCH_use_fp16_qk_reduction(use_fp16_qk_reduction, USE_FP16_QK_REDUCTION, {
                using Params = BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
                using AttentionVariant = DefaultAttention<
                    /*use_custom_mask=*/(MASK_MODE == MaskMode::kCustom),
                    /*use_sliding_window=*/false,
                    /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
                Params params(q, k, v, /*custom_mask=*/nullptr, qo_indptr, kv_indptr,
                              /*mask_indptr=*/nullptr, q_rope_offset, k_rope_offset, o, lse,
                              /*alibi_slopes=*/nullptr, num_qo_heads, num_kv_heads, qo_stride_n,
                              qo_stride_h, kv_stride_n, kv_stride_h, /*window_left=*/-1,
                              /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta);
                params.request_indices = handler->GetRequestIndices<IdType>();
                params.qo_tile_indices = handler->GetQOTileIndices<IdType>();
                params.kv_tile_indices = handler->GetKVTileIndices<IdType>();
                params.o_indptr = handler->GetOIndptr<IdType>();
                params.kv_chunk_size_ptr = handler->GetKVChunkSizePtr<IdType>();
                params.merge_indptr = handler->GetMergeIndptr<IdType>();
                params.block_valid_mask = handler->GetBlockValidMask();
                params.max_total_num_rows = plan_info.total_num_rows;
                params.total_num_rows = handler->GetTotalNumRows();
                params.padded_batch_size = plan_info.padded_batch_size;

                DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
                  BatchPrefillWithRaggedKVCacheDispatched<CTA_TILE_Q, HEAD_DIM, POS_ENCODING_MODE,
                                                          USE_FP16_QK_REDUCTION, MASK_MODE,
                                                          AttentionVariant>(
                      params, handler->GetTmpV<DTypeO>(), handler->GetTmpS(), stream);
                });
              })})})});
  return cudaSuccess;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeQ* q, IdType* qo_indptr, IdType* q_rope_offset,
    paged_kv_t<DTypeKV, IdType> paged_kv, DTypeO* o, float* lse, uint32_t num_qo_heads,
    bool causal = true, PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool use_fp16_qk_reduction = false, std::optional<float> maybe_sm_scale = std::nullopt,
    float rope_scale = 1.f, float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(paged_kv.head_dim)));
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  auto plan_info = handler->GetPlanInfo();
  DISPATCH_head_dim(
      head_dim, HEAD_DIM,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_pos_encoding_mode(
              pos_encoding_mode, POS_ENCODING_MODE,
              {DISPATCH_use_fp16_qk_reduction(use_fp16_qk_reduction, USE_FP16_QK_REDUCTION, {
                using Params = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
                using AttentionVariant = DefaultAttention<
                    /*use_custom_mask=*/(MASK_MODE == MaskMode::kCustom),
                    /*use_sliding_window=*/false,
                    /*use_logits_soft_cap=*/false,
                    /*use_alibi=*/false>;
                Params params(q, paged_kv, /*custom_mask=*/nullptr, qo_indptr,
                              /*mask_indptr=*/nullptr, q_rope_offset, o, lse,
                              /*alibi_slopes=*/nullptr, num_qo_heads,
                              /*q_stride_n*/ num_qo_heads * HEAD_DIM, /*q_stride_h*/ HEAD_DIM,
                              /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale, rope_scale,
                              rope_theta);
                params.request_indices = handler->GetRequestIndices<IdType>();
                params.qo_tile_indices = handler->GetQOTileIndices<IdType>();
                params.kv_tile_indices = handler->GetKVTileIndices<IdType>();
                params.o_indptr = handler->GetOIndptr<IdType>();
                params.kv_chunk_size_ptr = handler->GetKVChunkSizePtr<IdType>();
                params.merge_indptr = handler->GetMergeIndptr<IdType>();
                params.block_valid_mask = handler->GetBlockValidMask();
                params.max_total_num_rows = plan_info.total_num_rows;
                params.total_num_rows = handler->GetTotalNumRows();
                params.padded_batch_size = plan_info.padded_batch_size;
                DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
                  return BatchPrefillWithPagedKVCacheDispatched<
                      CTA_TILE_Q, HEAD_DIM, POS_ENCODING_MODE, USE_FP16_QK_REDUCTION, MASK_MODE,
                      AttentionVariant>(params, handler->GetTmpV<DTypeO>(), handler->GetTmpS(),
                                        stream);
                })
              })})})});
  return cudaSuccess;
}

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t SingleDecodeWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                              cudaStream_t stream);

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t SingleDecodeWithKVCache(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, DTypeO* tmp,
                                    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t seq_len,
                                    uint32_t head_dim, QKVLayout kv_layout = QKVLayout::kNHD,
                                    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                    std::optional<float> maybe_sm_scale = std::nullopt,
                                    float rope_scale = 1.f, float rope_theta = 1e4,
                                    cudaStream_t stream = nullptr) {
  float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  DISPATCH_head_dim(
      head_dim, HEAD_DIM, {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
        using Params = SingleDecodeParams<DTypeQ, DTypeKV, DTypeO>;
        using AttentionVariant = DefaultAttention<
            /*use_custom_mask=*/false, /*use_sliding_window=*/false,
            /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
        Params params(q, k, v, o, /*alibi_slopes=*/nullptr, seq_len, num_qo_heads, num_kv_heads,
                      kv_layout, head_dim, /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale,
                      rope_scale, rope_theta);

        SingleDecodeWithKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE, AttentionVariant>(
            params, tmp, stream);
      })});
  return cudaSuccess;
}

/*!
 * \brief Wrapper of BatchDecodeWithPagedKVCache function, and caches the temporary buffer
 *   for cooperative kernels.
 * \tparam DTypeQ The data type of query tensor.
 * \tparam DTypeKV The data type of key-value tensor.
 * \tparam DTypeO The data type of output tensor.
 * \tparam IdType The data type of index tensor.
 * \param handler The handler for the batch decode forward request.
 * \param q The input tensor.
 * \param paged_kv The paged key-value tensor.
 * \param o The output tensor.
 * \param lse The logsumexp values.
 * \param num_qo_heads The number of heads.
 * \param pos_encoding_mode The positional encoding mode.
 * \param rope_scale The scale of rope.
 * \param rope_theta The theta of rope.
 * \param stream The CUDA stream.
 */
template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWrapper(
    BatchDecodeHandler* handler, DTypeQ* q, IdType* q_rope_offset,
    paged_kv_t<DTypeKV, IdType> paged_kv, DTypeO* o, float* lse, uint32_t num_qo_heads,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    std::optional<float> maybe_sm_scale = std::nullopt, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(paged_kv.head_dim)));
  const uint32_t num_kv_heads = paged_kv.num_heads;
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  DISPATCH_head_dim(
      paged_kv.head_dim, HEAD_DIM,
      {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
        using Params = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
        using AttentionVariant = DefaultAttention<
            /*use_custom_mask=*/false, /*use_sliding_window=*/false,
            /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
        Params params(q, q_rope_offset, paged_kv, o, lse, /*alibi_slopes=*/nullptr, num_qo_heads,
                      /*q_stride_n*/ num_qo_heads * HEAD_DIM, /*q_stride_h*/ HEAD_DIM,
                      /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale, rope_scale,
                      rope_theta);
        params.request_indices = handler->GetRequestIndices<IdType>();
        params.kv_tile_indices = handler->GetKVTileIndices<IdType>();
        params.o_indptr = handler->GetOIndptr<IdType>();
        params.kv_chunk_size_ptr = handler->GetKVChunkSizePtr<IdType>();
        params.block_valid_mask = handler->GetBlockValidMask();
        params.padded_batch_size = handler->GetPlanInfo().padded_batch_size;

        return BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, POS_ENCODING_MODE, AttentionVariant>(
            params, handler->GetTmpV<DTypeO>(), handler->GetTmpS(), stream);
      })});
  return cudaSuccess;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
cudaError_t BatchDecodeHandlerPlan(BatchDecodeHandler* handler, void* float_buffer,
                                   size_t float_workspace_size_in_bytes, void* int_buffer,
                                   size_t int_workspace_size_in_bytes, IdType* indptr_h,
                                   IdType* last_page_len_h, uint32_t batch_size,
                                   uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                   uint32_t page_size, PosEncodingMode pos_encoding_mode) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    FLASHINFER_ERROR(err_msg.str());
  }
  DISPATCH_head_dim(head_dim, HEAD_DIM, {
    DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
      DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
        return handler->PlanDispatched<GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE, DTypeQ, DTypeKV,
                                       DTypeO, IdType>(
            float_buffer, float_workspace_size_in_bytes, int_buffer, int_workspace_size_in_bytes,
            indptr_h, last_page_len_h, batch_size, num_qo_heads, page_size);
      });
    });
  });
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
cudaError_t BatchDecodeWithPagedKVCacheWrapperMLA(
    BatchDecodeHandler* handler, DTypeQ* q_nope, DTypeQ* q_pe, IdType* q_rope_offset,
    paged_kv_mla_t<DTypeKV, IdType> paged_kv, DTypeO* o, float* lse, uint32_t num_qo_heads,
    float sm_scale, float rope_scale = 1.f, float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  DISPATCH_head_dim(paged_kv.head_dim_ckv, HEAD_DIM_CKV, {
    // fixme: head_dim_ckv(kv_lora_rank) is 8 times the size of head_dim_kpe(qk_rope_head_dim) for
    // all MLA model (DeepSeek-V2-Lite, DeepSeek-V2.5, MiniCPM3) at the time Oct.2024
    constexpr auto HEAD_DIM_KPE = HEAD_DIM_CKV / 8;
    using Params = BatchDecodeParamsMLA<DTypeQ, DTypeKV, DTypeO, IdType>;
    using AttentionVariant = DefaultAttention<
        /*use_custom_mask=*/false, /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
    Params params(q_nope, q_pe, q_rope_offset, paged_kv, o, lse, num_qo_heads,
                  /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta);
    params.request_indices = handler->GetRequestIndices<IdType>();
    params.kv_tile_indices = handler->GetKVTileIndices<IdType>();
    params.o_indptr = handler->GetOIndptr<IdType>();
    params.kv_chunk_size_ptr = handler->GetKVChunkSizePtr<IdType>();
    params.block_valid_mask = handler->GetBlockValidMask();
    params.padded_batch_size = handler->GetPlanInfo().padded_batch_size;

    return BatchDecodeWithPagedKVCacheDispatchedMLA<HEAD_DIM_CKV, HEAD_DIM_KPE, AttentionVariant>(
        params, handler->GetTmpV<DTypeO>(), handler->GetTmpS(), stream);
  });
  return cudaSuccess;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
cudaError_t BatchDecodeHandlerPlanMLA(BatchDecodeHandler* handler, void* float_buffer,
                                      size_t float_workspace_size_in_bytes, void* int_buffer,
                                      size_t int_workspace_size_in_bytes, IdType* indptr_h,
                                      IdType* last_page_len_h, uint32_t batch_size,
                                      uint32_t num_qo_heads, uint32_t head_dim_ckv,
                                      uint32_t page_size) {
  DISPATCH_head_dim(head_dim_ckv, HEAD_DIM_CKV, {
    // fixme: head_dim_ckv(kv_lora_rank) is 8 times the size of head_dim_kpe(qk_rope_head_dim) for
    // all MLA model (DeepSeek-V2-Lite, DeepSeek-V2.5, MiniCPM3) at the time Oct.2024
    constexpr auto HEAD_DIM_KPE = HEAD_DIM_CKV / 8;
    return handler->PlanDispatchedMLA<HEAD_DIM_CKV, HEAD_DIM_KPE, DTypeQ, DTypeKV, DTypeO, IdType>(
        float_buffer, float_workspace_size_in_bytes, int_buffer, int_workspace_size_in_bytes,
        indptr_h, last_page_len_h, batch_size, num_qo_heads, page_size);
  });
}

}  // namespace flashinfer
