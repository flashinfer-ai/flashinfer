// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include "flashinfer/attention/generic/default_decode_params.cuh"
#include "flashinfer/attention/generic/scheduler.cuh"
#include "flashinfer/attention/generic/variants.cuh"

#include "gpu_iface/enums.hpp"
#include "gpu_iface/layout.cuh"

#include "utils_hip.h"

#include <optional>

namespace flashinfer
{
class BatchDecodeHandler
{
public:
    template <uint32_t GROUP_SIZE,
              uint32_t HEAD_DIM,
              PosEncodingMode POS_ENCODING_MODE,
              typename DTypeQ,
              typename DTypeKV,
              typename DTypeO,
              typename IdType>
    hipError_t PlanDispatched(void *float_buffer,
                              size_t float_workspace_size_in_bytes,
                              void *int_buffer,
                              size_t int_workspace_size_in_bytes,
                              IdType *indptr_h,
                              IdType *last_page_len_h,
                              uint32_t batch_size,
                              uint32_t num_qo_heads,
                              uint32_t page_size)
    {
        int_buffer_ = int_buffer;
        float_buffer_ = float_buffer;
        using Params = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
        using AttentionVariant = DefaultAttention<
            /*use_custom_mask=*/false, /*use_sliding_window=*/false,
            /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;

        auto work_estimation_func =
            BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
                GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE, AttentionVariant,
                Params>;
        return DecodePlan<HEAD_DIM, POS_ENCODING_MODE, AttentionVariant,
                          Params>(
            float_buffer, float_workspace_size_in_bytes, int_buffer,
            page_locked_buffer_, int_workspace_size_in_bytes, plan_info_,
            indptr_h, batch_size, num_qo_heads, page_size, cuda_graph_enabled_,
            stream_, work_estimation_func);
    }

    void UpdatePageLockedBufferSize(size_t int_workspace_size_in_bytes)
    {
        hipFreeHost(page_locked_buffer_);
        hipMallocHost(&page_locked_buffer_, int_workspace_size_in_bytes);
    }

    hipStream_t GetCUDAStream() const { return stream_; }

    void SetCUDAStream(hipStream_t stream) { stream_ = stream; }

    /*!
     * \brief Constructor of BatchDecodeHandler
     * \param enable_cuda_graph A boolean indicates whether to enable CUDA graph
     * \param batch_size If enable_cuda_graph is true, we must specify a fixed
     * batch_size
     */
    BatchDecodeHandler(bool enable_cuda_graph = false, uint32_t batch_size = 0)
        : cuda_graph_enabled_(enable_cuda_graph), stream_(nullptr)
    {
        hipMallocHost(&page_locked_buffer_, 8 * 1024 * 1024);
    }
    ~BatchDecodeHandler() { hipFreeHost(page_locked_buffer_); }

    bool IsCUDAGraphEnabled() const { return cuda_graph_enabled_; }

    DecodePlanInfo GetPlanInfo() const { return plan_info_; }

    template <typename IdType> IdType *GetRequestIndices()
    {
        return GetPtrFromBaseOffset<IdType>(int_buffer_,
                                            plan_info_.request_indices_offset);
    }

    template <typename IdType> IdType *GetKVTileIndices()
    {
        return GetPtrFromBaseOffset<IdType>(int_buffer_,
                                            plan_info_.kv_tile_indices_offset);
    }

    template <typename IdType> IdType *GetOIndptr()
    {
        return GetPtrFromBaseOffset<IdType>(int_buffer_,
                                            plan_info_.o_indptr_offset);
    }

    template <typename IdType> IdType *GetKVChunkSizePtr()
    {
        return GetPtrFromBaseOffset<IdType>(
            int_buffer_, plan_info_.kv_chunk_size_ptr_offset);
    }

    template <typename DTypeO> DTypeO *GetTmpV()
    {
        if (plan_info_.split_kv) {
            return GetPtrFromBaseOffset<DTypeO>(float_buffer_,
                                                plan_info_.v_offset);
        }
        return nullptr;
    }

    float *GetTmpS()
    {
        if (plan_info_.split_kv) {
            return GetPtrFromBaseOffset<float>(float_buffer_,
                                               plan_info_.s_offset);
        }
        return nullptr;
    }

    bool *GetBlockValidMask()
    {
        if (plan_info_.split_kv && plan_info_.enable_cuda_graph) {
            return GetPtrFromBaseOffset<bool>(
                int_buffer_, plan_info_.block_valid_mask_offset);
        }
        return nullptr;
    }

protected:
    void *page_locked_buffer_;
    void *int_buffer_;
    void *float_buffer_;
    DecodePlanInfo plan_info_;
    bool cuda_graph_enabled_;
    hipStream_t stream_;
};

/*!
 * \brief Wrapper of BatchDecodeWithPagedKVCache function, and caches the
 * temporary buffer for cooperative kernels.
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
hipError_t BatchDecodeWithPagedKVCacheWrapper(
    BatchDecodeHandler *handler,
    DTypeQ *q,
    IdType *q_rope_offset,
    paged_kv_t<DTypeKV, IdType> paged_kv,
    DTypeO *o,
    float *lse,
    uint32_t num_qo_heads,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    std::optional<float> maybe_sm_scale = std::nullopt,
    float rope_scale = 1.f,
    float rope_theta = 1e4,
    hipStream_t stream = nullptr)
{
    float sm_scale =
        maybe_sm_scale.value_or(1.f / std::sqrt(float(paged_kv.head_dim)));
    const uint32_t num_kv_heads = paged_kv.num_heads;
    if (num_qo_heads % num_kv_heads != 0) {
        std::ostringstream err_msg;
        err_msg << "num_qo_heads " << num_qo_heads
                << " is not a multiple of num_kv_heads " << num_kv_heads;
        FLASHINFER_ERROR(err_msg.str());
    }

    DISPATCH_head_dim(
        paged_kv.head_dim, HEAD_DIM,
        {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
            using Params = BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
            using AttentionVariant = DefaultAttention<
                /*use_custom_mask=*/false, /*use_sliding_window=*/false,
                /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
            Params params(q, q_rope_offset, paged_kv, o, lse,
                          /*alibi_slopes=*/nullptr, num_qo_heads,
                          /*q_stride_n*/ num_qo_heads * HEAD_DIM,
                          /*q_stride_h*/ HEAD_DIM,
                          /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale,
                          rope_scale, rope_theta);
            params.request_indices = handler->GetRequestIndices<IdType>();
            params.kv_tile_indices = handler->GetKVTileIndices<IdType>();
            params.o_indptr = handler->GetOIndptr<IdType>();
            params.kv_chunk_size_ptr = handler->GetKVChunkSizePtr<IdType>();
            params.block_valid_mask = handler->GetBlockValidMask();
            params.padded_batch_size = handler->GetPlanInfo().padded_batch_size;

            return BatchDecodeWithPagedKVCacheDispatched<
                HEAD_DIM, POS_ENCODING_MODE, AttentionVariant>(
                params, handler->GetTmpV<DTypeO>(), handler->GetTmpS(), stream);
        })});
    return hipSuccess;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
hipError_t BatchDecodeHandlerPlan(BatchDecodeHandler *handler,
                                  void *float_buffer,
                                  size_t float_workspace_size_in_bytes,
                                  void *int_buffer,
                                  size_t int_workspace_size_in_bytes,
                                  IdType *indptr_h,
                                  IdType *last_page_len_h,
                                  uint32_t batch_size,
                                  uint32_t num_qo_heads,
                                  uint32_t num_kv_heads,
                                  uint32_t head_dim,
                                  uint32_t page_size,
                                  PosEncodingMode pos_encoding_mode)
{
    if (num_qo_heads % num_kv_heads != 0) {
        std::ostringstream err_msg;
        err_msg << "num_qo_heads " << num_qo_heads
                << " should be divisible by num_kv_heads " << num_kv_heads;
        FLASHINFER_ERROR(err_msg.str());
    }
    DISPATCH_head_dim(head_dim, HEAD_DIM, {
        DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
            DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
                return handler
                    ->PlanDispatched<GROUP_SIZE, HEAD_DIM, POS_ENCODING_MODE,
                                     DTypeQ, DTypeKV, DTypeO, IdType>(
                        float_buffer, float_workspace_size_in_bytes, int_buffer,
                        int_workspace_size_in_bytes, indptr_h, last_page_len_h,
                        batch_size, num_qo_heads, page_size);
            });
        });
    });

    return hipSuccess;
}

} // namespace flashinfer
