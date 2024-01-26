/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include <dlpack/dlpack.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <flashinfer.cuh>

using tvm::runtime::Array;
using tvm::runtime::DataType;
using tvm::runtime::NDArray;
using tvm::runtime::ShapeTuple;
using namespace flashinfer;

#define SWITCH_TVM_CUDA_DTYPE(dl_dtype, cuda_dtype, ...)     \
  if (dl_dtype.code == kDLFloat && dl_dtype.bits == 16) {    \
    using cuda_dtype = half;                                 \
    __VA_ARGS__                                              \
  } else {                                                   \
    LOG(FATAL) << "Unsupported data type " << dl_dtype.code; \
  }

#define SWITCH_TVM_CUDA_IDTYPE(dl_dtype, cuda_dtype, ...)    \
  if (dl_dtype.code == kDLInt && dl_dtype.bits == 32) {      \
    using cuda_dtype = int32_t;                              \
    __VA_ARGS__                                              \
  } else {                                                   \
    LOG(FATAL) << "Unsupported data type " << dl_dtype.code; \
  }

/*!
 * \brief The SinglePrefillWithKVCache function with some parameters fixed at compile time
 *   to accelerate the dispatching.
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t _SinglePrefillWithKVCacheNoLSE(
    DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o, float* tmp, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len, uint32_t head_dim, bool causal = true,
    QKVLayout kv_layout = QKVLayout::kNHD, RotaryMode rotary_mode = RotaryMode::kNone,
    bool allow_fp16_qk_reduction = false, float rope_scale = 1.f, float rope_theta = 1e4,
    cudaStream_t stream = nullptr) {
  CHECK(head_dim == 128) << "The head dimension must be 128";
  CHECK(kv_layout == QKVLayout::kNHD) << "The KV layout must be NHD";
  const uint32_t group_size = num_qo_heads / num_kv_heads;

  SWITCH_ALLOW_FP16_QK_REDUCTION(
      allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION,
      {SWITCH_GQA_GROUP_SIZE(
          group_size, GROUP_SIZE,
          {SWITCH_CAUSAL(causal, CAUSAL, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
                           SinglePrefillWithKVCacheDispatched<
                               GROUP_SIZE, /*head_dim=*/128, /*layout=*/QKVLayout::kNHD,
                               ROTARY_MODE, ALLOW_FP16_QK_REDUCTION, CAUSAL>(
                               q, k, v, o, tmp, /*lse=*/nullptr, num_kv_heads, qo_len, kv_len,
                               rope_scale, rope_theta, stream);
                         })})})});
  return cudaSuccess;
}

int _FlashInferSinglePrefillWithKVCache(DLTensor* q, DLTensor* k, DLTensor* v, DLTensor* tmp,
                                        bool causal, int64_t kv_layout, int64_t rotary_mode,
                                        bool allow_fp16_qk_reduction, double rope_scale,
                                        double rope_theta, DLTensor* o) {
  // `tmp` is user-provided scratch space of at least 16MB, e.g. 4 * 1024 * 1024 float32.
  CHECK_EQ(q->device.device_type, kDLCUDA) << "The device of q matrix must be CUDA.";
  CHECK_EQ(k->device.device_type, kDLCUDA) << "The device of k matrix must be CUDA.";
  CHECK_EQ(v->device.device_type, kDLCUDA) << "The device of v matrix must be CUDA.";
  CHECK_EQ(o->device.device_type, kDLCUDA) << "The device of o matrix must be CUDA.";

  size_t dev_id = q->device.device_id;
  CHECK_EQ(k->device.device_id, dev_id) << "The device id of q and k matrix doesn't match.";
  CHECK_EQ(v->device.device_id, dev_id) << "The device id of q and v matrix doesn't match.";
  CHECK_EQ(o->device.device_id, dev_id) << "The device id of q and o matrix doesn't match.";

  CHECK_GE(q->ndim, 3);
  size_t qo_len = q->shape[q->ndim - 3];
  size_t num_qo_heads = q->shape[q->ndim - 2];
  size_t head_dim = q->shape[q->ndim - 1];

  CHECK_GE(k->ndim, 3);
  size_t kv_len = k->shape[k->ndim - 3];
  size_t num_kv_heads = k->shape[k->ndim - 2];
  CHECK_EQ(head_dim, k->shape[k->ndim - 1]);

  CHECK_GE(v->ndim, 3);
  CHECK_EQ(kv_len, v->shape[v->ndim - 3]);
  CHECK_EQ(num_kv_heads, v->shape[v->ndim - 2]);
  CHECK_EQ(head_dim, v->shape[v->ndim - 1]);

  CHECK_GE(o->ndim, 2);
  CHECK_EQ(qo_len, o->shape[o->ndim - 2]);
  CHECK_EQ(num_qo_heads * head_dim, o->shape[o->ndim - 1]);

  CHECK(q->dtype.lanes == 1 && k->dtype.lanes == 1 && v->dtype.lanes == 1);
  CHECK(q->dtype.bits == k->dtype.bits && q->dtype.code == k->dtype.code);
  CHECK(q->dtype.bits == v->dtype.bits && q->dtype.code == v->dtype.code);

  SWITCH_TVM_CUDA_DTYPE(
      q->dtype, dtype_in, {SWITCH_TVM_CUDA_DTYPE(o->dtype, dtype_out, {
        cudaError_t status = _SinglePrefillWithKVCacheNoLSE(
            (dtype_in*)q->data, (dtype_in*)k->data, (dtype_in*)v->data, (dtype_out*)o->data,
            (float*)tmp->data, num_qo_heads, num_kv_heads, qo_len, kv_len, head_dim, causal,
            QKVLayout(kv_layout), RotaryMode(rotary_mode), allow_fp16_qk_reduction, rope_scale,
            rope_theta, 0);
        if (status != cudaSuccess) {
          LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
        }
      })});
  return 0;
}

int _FlashInferSingleDecodeWithKVCache(DLTensor* q, DLTensor* k, DLTensor* v, DLTensor* tmp,
                                       int64_t kv_layout, int64_t rotary_mode, double rope_scale,
                                       double rope_theta, DLTensor* o) {
  // `tmp` is user-provided scratch space of at least 16MB, e.g. 4 * 1024 * 1024 float32.
  CHECK_EQ(q->device.device_type, kDLCUDA) << "The device of q matrix must be CUDA.";
  CHECK_EQ(k->device.device_type, kDLCUDA) << "The device of k matrix must be CUDA.";
  CHECK_EQ(v->device.device_type, kDLCUDA) << "The device of v matrix must be CUDA.";
  CHECK_EQ(o->device.device_type, kDLCUDA) << "The device of o matrix must be CUDA.";

  size_t dev_id = q->device.device_id;
  CHECK_EQ(k->device.device_id, dev_id) << "The device id of q and k matrix doesn't match.";
  CHECK_EQ(v->device.device_id, dev_id) << "The device id of q and v matrix doesn't match.";
  CHECK_EQ(o->device.device_id, dev_id) << "The device id of q and o matrix doesn't match.";

  CHECK_GE(q->ndim, 2);
  size_t num_qo_heads = q->shape[q->ndim - 2];
  size_t head_dim = q->shape[q->ndim - 1];

  CHECK_GE(k->ndim, 3);
  size_t seq_len = k->shape[k->ndim - 3];
  size_t num_kv_heads = k->shape[k->ndim - 2];
  CHECK_EQ(head_dim, k->shape[k->ndim - 1]);

  CHECK_GE(v->ndim, 3);
  CHECK_EQ(seq_len, v->shape[v->ndim - 3]);
  CHECK_EQ(num_kv_heads, v->shape[v->ndim - 2]);
  CHECK_EQ(head_dim, v->shape[v->ndim - 1]);

  CHECK_GE(o->ndim, 1);
  CHECK_EQ(num_qo_heads * head_dim, o->shape[o->ndim - 1]);

  CHECK(q->dtype.lanes == 1 && k->dtype.lanes == 1 && v->dtype.lanes == 1);
  CHECK(q->dtype.bits == k->dtype.bits && q->dtype.code == k->dtype.code);
  CHECK(q->dtype.bits == v->dtype.bits && q->dtype.code == v->dtype.code);

  SWITCH_TVM_CUDA_DTYPE(
      q->dtype, dtype_in, {SWITCH_TVM_CUDA_DTYPE(o->dtype, dtype_out, {
        cudaError_t status = SingleDecodeWithKVCache(
            (dtype_in*)q->data, (dtype_in*)k->data, (dtype_in*)v->data, (dtype_out*)o->data,
            (dtype_out*)tmp->data, num_qo_heads, num_kv_heads, seq_len, head_dim,
            QKVLayout(kv_layout), RotaryMode(rotary_mode), rope_scale, rope_theta, 0);
        if (status != cudaSuccess) {
          LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
        }
      })});
  return 0;
}

constexpr uint32_t max_num_handlers = 8;
thread_local BatchPrefillHandler batch_prefill_paged_kv_handlers[max_num_handlers];
thread_local BatchPrefillHandler batch_prefill_ragged_kv_handler;

/*!
 * \brief The BatchPrefillWithKVCacheWrapper function with some parameters fixed at compile time
 *    to accelerate the dispatching.
 */
template <PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t _BatchPrefillWithPagedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* lse,
    uint32_t num_qo_heads, bool causal = true, RotaryMode rotary_mode = RotaryMode::kNone,
    bool allow_fp16_qk_reduction = false, float rope_scale = 1.f, float rope_theta = 1e4,
    cudaStream_t stream = nullptr) {
  CHECK(lse != nullptr) << "The lse buffer must be provided";
  CHECK(allow_fp16_qk_reduction == false) << "The fp16 qk reduction is not supported";
  CHECK(paged_kv.head_dim == 128) << "The head dimension must be 128";
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  const uint32_t group_size = num_qo_heads / num_kv_heads;
  SWITCH_GQA_GROUP_SIZE(
      group_size, GROUP_SIZE,
      {SWITCH_CAUSAL(causal, CAUSAL, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
                       return BatchPrefillWithPagedKVCacheWrapperDispatched<
                           page_storage, kv_layout, GROUP_SIZE, /*head_dim=*/128, ROTARY_MODE,
                           /*allow_fp16_qk_reduction=*/false, CAUSAL, DTypeIn, DTypeOut, IdType>(
                           handler, q, qo_indptr, paged_kv, o, lse, num_qo_heads, rope_scale,
                           rope_theta, stream);
                     })})});
  return cudaSuccess;
}

void _FlashInferAttentionPrefillWithPagedKVCache(int64_t handler_id, DLTensor* q_data,
                                                 DLTensor* qo_indptr,          //
                                                 DLTensor* pages,              //
                                                 DLTensor* page_table_indptr,  //
                                                 DLTensor* page_table_values,  //
                                                 DLTensor* last_page_len,      //
                                                 DLTensor* output,             //
                                                 DLTensor* lse,                //
                                                 int64_t causal = 1,           //
                                                 int64_t rotary_mode = 0,      //
                                                 double rope_scale = 1.0f,     //
                                                 double rope_theta = 1e4) {
  CHECK(handler_id < max_num_handlers) << "The handler id must be less than " << max_num_handlers;
  CHECK_EQ(q_data->device.device_type, kDLCUDA) << "The device of q_data must be CUDA.";
  CHECK_EQ(pages->device.device_type, kDLCUDA) << "The device of kv pages must be CUDA.";
  CHECK_EQ(page_table_indptr->device.device_type, kDLCUDA)
      << "The device of page_table_indptr matrix must be CUDA.";
  CHECK_EQ(page_table_values->device.device_type, kDLCUDA)
      << "The device of page_table_values matrix must be CUDA.";
  CHECK_EQ(last_page_len->device.device_type, kDLCUDA)
      << "The device of last_page_len matrix must be CUDA.";
  CHECK_EQ(qo_indptr->device.device_type, kDLCUDA)
      << "The device of qo_indptr matrix must be CUDA.";
  CHECK_EQ(output->device.device_type, kDLCUDA) << "The device of output must be CUDA.";

  int32_t dev_id = q_data->device.device_id;
  CHECK_EQ(pages->device.device_id, dev_id);
  CHECK_EQ(page_table_indptr->device.device_id, dev_id);
  CHECK_EQ(page_table_values->device.device_id, dev_id);
  CHECK_EQ(last_page_len->device.device_id, dev_id);
  CHECK_EQ(qo_indptr->device.device_id, dev_id);
  CHECK_EQ(output->device.device_id, dev_id);

  CHECK(q_data->dtype.lanes == 1 && pages->dtype.lanes == 1 && output->dtype.lanes == 1);
  CHECK(q_data->dtype.bits == pages->dtype.bits && q_data->dtype.code == pages->dtype.code);
  CHECK(page_table_indptr->dtype.lanes == 1 && page_table_values->dtype.lanes == 1 &&
        last_page_len->dtype.lanes == 1 && qo_indptr->dtype.lanes == 1);
  CHECK(page_table_indptr->dtype.bits == page_table_values->dtype.bits &&
        page_table_indptr->dtype.bits == last_page_len->dtype.bits &&
        page_table_indptr->dtype.bits == qo_indptr->dtype.bits &&
        page_table_indptr->dtype.code == page_table_values->dtype.code &&
        page_table_indptr->dtype.code == last_page_len->dtype.code &&
        page_table_indptr->dtype.code == qo_indptr->dtype.code);

  CHECK_EQ(pages->ndim, 5);
  CHECK_EQ(pages->shape[1], 2);
  int64_t nhead_kv = pages->shape[2];
  int64_t nhead_qo = q_data->shape[1];
  int64_t nfeat = pages->shape[4];
  int64_t page_size = pages->shape[3];

  CHECK_EQ(last_page_len->ndim, 1);
  int64_t num_total_seqs = last_page_len->shape[0];

  CHECK_EQ(qo_indptr->ndim, 1);
  CHECK_EQ(qo_indptr->shape[0], num_total_seqs + 1);

  CHECK_EQ(page_table_indptr->ndim, 1);
  CHECK_EQ(page_table_indptr->shape[0], num_total_seqs + 1);
  CHECK_EQ(page_table_values->ndim, 1);

  CHECK_EQ(q_data->ndim, 3);
  CHECK_EQ(output->ndim, 3);
  CHECK_EQ(q_data->shape[2], nfeat);
  CHECK_EQ(output->shape[1], nhead_qo);
  CHECK_EQ(output->shape[2], nfeat);

  constexpr PageStorage page_storage = PageStorage::kIndices;
  constexpr QKVLayout kv_layout = QKVLayout::kHND;

  SWITCH_TVM_CUDA_DTYPE(
      pages->dtype, dtype_in,
      {SWITCH_TVM_CUDA_DTYPE(
          output->dtype, dtype_out, {SWITCH_TVM_CUDA_IDTYPE(page_table_values->dtype, dtype_idx, {
            paged_kv_t<page_storage, kv_layout, dtype_in, dtype_idx> cache(
                nhead_kv, page_size, nfeat, num_total_seqs, static_cast<dtype_in*>(pages->data),
                static_cast<dtype_idx*>(page_table_values->data),
                static_cast<dtype_idx*>(page_table_indptr->data),
                static_cast<dtype_idx*>(last_page_len->data));
            cudaError_t status =
                _BatchPrefillWithPagedKVCacheWrapper<page_storage, kv_layout, dtype_in, dtype_out,
                                                     dtype_idx>(
                    &batch_prefill_paged_kv_handlers[handler_id],
                    static_cast<dtype_in*>(q_data->data), static_cast<dtype_idx*>(qo_indptr->data),
                    cache, static_cast<dtype_out*>(output->data),
                    /*lse=*/static_cast<float*>(lse->data), nhead_qo,
                    /*causal=*/causal, RotaryMode(rotary_mode), /*allow_fp16_qk_reduction=*/false,
                    rope_scale, rope_theta, 0);
            if (status != cudaSuccess) {
              LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
            }
          })})});
}

void _FlashInferAttentionPrefillWithPagedKVCacheBeginForward(
    int64_t handler_idx, DLTensor* workspace_buffer, DLTensor* qo_indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads) {
  CHECK_EQ(workspace_buffer->ndim, 1) << "The workspace buffer must be a 1-D tensor";
  size_t workspace_size_in_bytes = workspace_buffer->shape[0] * workspace_buffer->dtype.bits / 8;
  CHECK(handler_idx < max_num_handlers) << "The handler id must be less than " << max_num_handlers;
  SWITCH_TVM_CUDA_IDTYPE(qo_indptr->dtype, dtype_idx, {
    cudaError_t status = batch_prefill_paged_kv_handlers[handler_idx].BeginForward(
        static_cast<void*>(workspace_buffer->data), workspace_size_in_bytes,
        static_cast<dtype_idx*>(qo_indptr->data), batch_size, num_qo_heads, num_kv_heads);
    if (status != cudaSuccess) {
      LOG(FATAL) << "FlashInfer prefill BeginForward error " << cudaGetErrorString(status);
    }
  });
}

void _FlashInferAttentionPrefillWithPagedKVCacheEndForward(int64_t handler_idx) {
  CHECK(handler_idx < max_num_handlers) << "The handler id must be less than " << max_num_handlers;
  batch_prefill_paged_kv_handlers[handler_idx].EndForward();
}

// Creates a pool of handlers with a fixed size to independently handle decoding forward passes.
thread_local BatchDecodeHandler batch_decode_handlers[max_num_handlers];

void _FlashInferAttentionDecodeWithPagedKVCache(int64_t handler_id, DLTensor* q_data,
                                                DLTensor* pages,
                                                DLTensor* page_table_indptr,  //
                                                DLTensor* page_table_values,  //
                                                DLTensor* last_page_len,      //
                                                DLTensor* output,             //
                                                DLTensor* lse,                //
                                                int64_t rotary_mode = 0,      //
                                                double rope_scale = 1.0f,     //
                                                double rope_theta = 1e4) {
  CHECK_LT(handler_id, max_num_handlers) << "The handler id must be less than " << max_num_handlers;
  CHECK_EQ(q_data->device.device_type, kDLCUDA) << "The device of q_data must be CUDA.";
  CHECK_EQ(pages->device.device_type, kDLCUDA) << "The device of kv pages must be CUDA.";
  CHECK_EQ(page_table_indptr->device.device_type, kDLCUDA)
      << "The device of page_table_indptr matrix must be CUDA.";
  CHECK_EQ(page_table_values->device.device_type, kDLCUDA)
      << "The device of page_table_values matrix must be CUDA.";
  CHECK_EQ(last_page_len->device.device_type, kDLCUDA)
      << "The device of last_page_len matrix must be CUDA.";
  CHECK_EQ(output->device.device_type, kDLCUDA) << "The device of output must be CUDA.";

  int32_t dev_id = q_data->device.device_id;
  CHECK_EQ(pages->device.device_id, dev_id);
  CHECK_EQ(page_table_indptr->device.device_id, dev_id);
  CHECK_EQ(page_table_values->device.device_id, dev_id);
  CHECK_EQ(last_page_len->device.device_id, dev_id);
  CHECK_EQ(output->device.device_id, dev_id);

  CHECK(q_data->dtype.lanes == 1 && pages->dtype.lanes == 1 && output->dtype.lanes == 1);
  CHECK(q_data->dtype.bits == pages->dtype.bits && q_data->dtype.code == pages->dtype.code);
  CHECK(page_table_indptr->dtype.lanes == 1 && page_table_values->dtype.lanes == 1 &&
        last_page_len->dtype.lanes == 1);
  CHECK(page_table_indptr->dtype.bits == page_table_values->dtype.bits &&
        page_table_indptr->dtype.bits == last_page_len->dtype.bits &&
        page_table_indptr->dtype.code == page_table_values->dtype.code &&
        page_table_indptr->dtype.code == last_page_len->dtype.code);

  CHECK_EQ(pages->ndim, 5);
  CHECK_EQ(pages->shape[1], 2);
  int64_t nhead_kv = pages->shape[2];
  int64_t nfeat = pages->shape[4];
  int64_t page_size = pages->shape[3];

  CHECK_EQ(last_page_len->ndim, 1);
  int64_t num_total_seqs = last_page_len->shape[0];

  CHECK_EQ(page_table_indptr->ndim, 1);
  CHECK_EQ(page_table_indptr->shape[0], num_total_seqs + 1);
  CHECK_EQ(page_table_values->ndim, 1);

  CHECK_EQ(q_data->ndim, 3);
  CHECK_EQ(output->ndim, 3);
  CHECK_GE(q_data->shape[0], 1);
  CHECK_EQ(q_data->shape[0], output->shape[0]);
  CHECK_EQ(q_data->shape[2], nfeat);
  int64_t nhead_qo = q_data->shape[1];
  CHECK_EQ(output->shape[1], nhead_qo);
  CHECK_EQ(output->shape[2], nfeat);

  constexpr PageStorage page_storage = PageStorage::kIndices;
  constexpr QKVLayout kv_layout = QKVLayout::kHND;

  SWITCH_TVM_CUDA_DTYPE(
      pages->dtype, dtype_in,
      {SWITCH_TVM_CUDA_DTYPE(
          output->dtype, dtype_out, {SWITCH_TVM_CUDA_IDTYPE(page_table_values->dtype, dtype_idx, {
            paged_kv_t<page_storage, kv_layout, dtype_in, dtype_idx> cache(
                nhead_kv, page_size, nfeat, num_total_seqs, static_cast<dtype_in*>(pages->data),
                static_cast<dtype_idx*>(page_table_values->data),
                static_cast<dtype_idx*>(page_table_indptr->data),
                static_cast<dtype_idx*>(last_page_len->data));
            cudaError_t status = BatchDecodeWithPagedKVCacheWrapper<page_storage, kv_layout,
                                                                    dtype_in, dtype_out, dtype_idx>(
                &batch_decode_handlers[handler_id], static_cast<dtype_in*>(q_data->data), cache,
                static_cast<dtype_out*>(output->data),
                /*lse=*/static_cast<float*>(lse->data), nhead_qo, RotaryMode(rotary_mode),
                rope_scale, rope_theta, 0);
            if (status != cudaSuccess) {
              LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
            }
          })})});
}

void _FlashInferAttentionDecodeWithPagedKVCacheBeginForward(
    int64_t handler_idx, DLTensor* workspace_buffer, DLTensor* page_table_indptr,
    DLTensor* last_page_len, int64_t num_qo_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t page_size, int64_t rotary_mode) {
  CHECK_EQ(workspace_buffer->ndim, 1) << "The workspace buffer must be a 1-D tensor";
  size_t workspace_size_in_bytes = workspace_buffer->shape[0] * workspace_buffer->dtype.bits / 8;
  CHECK_LT(handler_idx, max_num_handlers)
      << "The handler id must be less than " << max_num_handlers;
  constexpr PageStorage page_storage = PageStorage::kIndices;
  constexpr QKVLayout kv_layout = QKVLayout::kHND;
  // NOTE(Zihao): here we presume the input data type is half, in the future we should
  //   leave a parameter for the input data type.
  using dtype_in = half;
  const uint32_t batch_size = page_table_indptr->shape[0] - 1;
  SWITCH_TVM_CUDA_IDTYPE(page_table_indptr->dtype, dtype_idx, {
    cudaError_t status =
        batch_decode_handlers[handler_idx]
            .BeginForward<page_storage, kv_layout, dtype_in, dtype_in, dtype_idx>(
                static_cast<void*>(workspace_buffer->data), workspace_size_in_bytes,
                static_cast<dtype_idx*>(page_table_indptr->data),
                static_cast<dtype_idx*>(last_page_len->data), batch_size, num_qo_heads,
                num_kv_heads, head_dim, page_size, RotaryMode(rotary_mode));
    if (status != cudaSuccess) {
      LOG(FATAL) << "FlashInfer decode BeginForward error " << cudaGetErrorString(status);
    }
  });
}

void _FlashInferAttentionDecodeWithPagedKVCacheEndForward(int64_t handler_id) {
  CHECK_LT(handler_id, max_num_handlers) << "The handler id must be less than " << max_num_handlers;
  batch_decode_handlers[handler_id].EndForward();
}

/*!
 * \brief The BatchPrefillWithRaggedKVCacheWrapper function with some parameters fixed at compile
 * time to accelerate the dispatching.
 */
template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t _BatchPrefillWithRaggedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeIn* q, IdType* qo_indptr, DTypeIn* k, DTypeIn* v,
    IdType* kv_indptr, DTypeOut* o, float* lse, const uint32_t batch_size,
    const uint32_t num_qo_heads, const uint32_t num_kv_heads, const uint32_t head_dim,
    bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
    RotaryMode rotary_mode = RotaryMode::kNone, bool allow_fp16_qk_reduction = false,
    const float rope_scale = 1.f, const float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  CHECK(lse != nullptr) << "The lse buffer must be provided";
  CHECK(head_dim == 128) << "The head dimension must be 128";
  CHECK(kv_layout == QKVLayout::kNHD) << "The layout must be NHD";
  CHECK(allow_fp16_qk_reduction == false) << "The fp16 qk reduction is not supported";
  SWITCH_GQA_GROUP_SIZE(
      num_qo_heads / num_kv_heads, GROUP_SIZE,
      {SWITCH_CAUSAL(causal, CAUSAL, {SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, {
                       return BatchPrefillWithRaggedKVCacheWrapperDispatched<
                           GROUP_SIZE, /*head_dim=*/128, /*layout=*/QKVLayout::kNHD, ROTARY_MODE,
                           /*allow_fp16_qk_reduction=*/false, CAUSAL, DTypeIn, DTypeOut, IdType>(
                           handler, q, qo_indptr, k, v, kv_indptr, o, lse, batch_size, num_kv_heads,
                           rope_scale, rope_theta, stream);
                     })})});
  return cudaSuccess;
}

void _FlashInferAttentionPrefillWithRaggedKVCache(DLTensor* q_data, DLTensor* qo_indptr,
                                                  DLTensor* k_data, DLTensor* v_data,
                                                  DLTensor* kv_indptr, DLTensor* output,
                                                  DLTensor* lse, int64_t causal = 1,
                                                  int64_t rotary_mode = 0, double rope_scale = 1.0f,
                                                  double rope_theta = 1e4) {
  CHECK_EQ(q_data->device.device_type, kDLCUDA) << "The device of q_data must be CUDA.";
  CHECK_EQ(qo_indptr->device.device_type, kDLCUDA) << "The device of qo_indptr must be CUDA.";
  CHECK_EQ(k_data->device.device_type, kDLCUDA) << "The device of k_data must be CUDA.";
  CHECK_EQ(v_data->device.device_type, kDLCUDA) << "The device of v_data must be CUDA.";
  CHECK_EQ(kv_indptr->device.device_type, kDLCUDA) << "The device of kv_indptr must be CUDA.";
  CHECK_EQ(output->device.device_type, kDLCUDA) << "The device of output must be CUDA.";

  int dev_id = q_data->device.device_id;
  CHECK_EQ(qo_indptr->device.device_id, dev_id);
  CHECK_EQ(k_data->device.device_id, dev_id);
  CHECK_EQ(v_data->device.device_id, dev_id);
  CHECK_EQ(kv_indptr->device.device_id, dev_id);
  CHECK_EQ(output->device.device_id, dev_id);
  CHECK_EQ(lse->device.device_id, dev_id);

  CHECK(q_data->dtype.lanes == 1 && qo_indptr->dtype.lanes == 1 && k_data->dtype.lanes == 1 &&
        v_data->dtype.lanes == 1 && kv_indptr->dtype.lanes == 1 && output->dtype.lanes == 1 &&
        lse->dtype.lanes == 1);
  CHECK(q_data->dtype.bits == k_data->dtype.bits && q_data->dtype.code == v_data->dtype.code);
  CHECK(qo_indptr->dtype.bits == kv_indptr->dtype.bits);
  CHECK(lse->dtype.bits == 32);
  CHECK(q_data->dtype.code == k_data->dtype.code && q_data->dtype.code == v_data->dtype.code);
  CHECK(qo_indptr->dtype.code == kv_indptr->dtype.code);
  CHECK(lse->dtype.code == kDLFloat);

  CHECK_EQ(q_data->ndim, 3);  // qo_nnz, nhead_qo, nfeat
  CHECK_EQ(output->ndim, 3);  // qo_nnz, nhead_qo, nfeat
  CHECK_EQ(lse->ndim, 2);     // qo_nnz, nhead_qo
  CHECK_EQ(k_data->ndim, 3);  // kv_nnz, nhead_kv, nfeat
  CHECK_EQ(v_data->ndim, 3);  // kv_nnz, nhead_kv, nfeat
  int64_t nhead_qo = q_data->shape[1];
  int64_t nfeat = q_data->shape[2];
  int64_t nhead_kv = k_data->shape[1];
  CHECK_EQ(output->shape[0], q_data->shape[0]);
  CHECK_EQ(output->shape[1], nhead_qo);
  CHECK_EQ(output->shape[2], nfeat);
  CHECK_EQ(lse->shape[0], q_data->shape[0]);
  CHECK_EQ(lse->shape[1], nhead_qo);
  CHECK_EQ(k_data->shape[2], nfeat);
  CHECK_EQ(v_data->shape[0], k_data->shape[0]);
  CHECK_EQ(v_data->shape[1], nhead_kv);
  CHECK_EQ(v_data->shape[2], nfeat);

  CHECK_EQ(qo_indptr->ndim, 1);
  CHECK_EQ(kv_indptr->ndim, 1);
  int64_t batch_size = qo_indptr->shape[0] - 1;
  CHECK_EQ(kv_indptr->shape[0], batch_size + 1);

  SWITCH_TVM_CUDA_DTYPE(
      q_data->dtype, dtype_in,
      {SWITCH_TVM_CUDA_DTYPE(
          output->dtype, dtype_out, {SWITCH_TVM_CUDA_IDTYPE(qo_indptr->dtype, dtype_idx, {
            cudaError_t status =
                _BatchPrefillWithRaggedKVCacheWrapper<dtype_in, dtype_out, dtype_idx>(
                    &batch_prefill_ragged_kv_handler, static_cast<dtype_in*>(q_data->data),
                    static_cast<dtype_idx*>(qo_indptr->data), static_cast<dtype_in*>(k_data->data),
                    static_cast<dtype_in*>(v_data->data), static_cast<dtype_idx*>(kv_indptr->data),
                    static_cast<dtype_out*>(output->data),
                    /*lse=*/static_cast<float*>(lse->data), batch_size, nhead_qo, nhead_kv, nfeat,
                    /*causal=*/bool(causal), QKVLayout::kNHD, RotaryMode(rotary_mode),
                    /*allow_fp16_qk_reduction=*/false, rope_scale, rope_theta, 0);
          })})})
}

void _FlashInferAttentionPrefillWithRaggedKVCacheBeginForward(DLTensor* workspace_buffer,
                                                              DLTensor* qo_indptr,
                                                              int64_t batch_size,
                                                              int64_t num_qo_heads,
                                                              int64_t num_kv_heads) {
  CHECK_EQ(workspace_buffer->ndim, 1) << "The workspace buffer must be a 1-D tensor";
  size_t workspace_size_in_bytes = workspace_buffer->shape[0] * workspace_buffer->dtype.bits / 8;

  SWITCH_TVM_CUDA_IDTYPE(qo_indptr->dtype, dtype_idx, {
    cudaError_t status = batch_prefill_ragged_kv_handler.BeginForward(
        static_cast<void*>(workspace_buffer->data), workspace_size_in_bytes,
        static_cast<dtype_idx*>(qo_indptr->data), batch_size, num_qo_heads, num_kv_heads);
    if (status != cudaSuccess) {
      LOG(FATAL) << "FlashInfer PrefillWithRaggedKVCache BeginForward error "
                 << cudaGetErrorString(status);
    }
  });
}

void _FlashInferAttentionPrefillWithRaggedKVCacheEndForward() {
  batch_prefill_ragged_kv_handler.EndForward();
}

void _FlashInferMergeState(DLTensor* v_a, DLTensor* s_a, DLTensor* v_b, DLTensor* s_b,
                           DLTensor* v_merged, DLTensor* s_merged) {
  CHECK_EQ(v_a->device.device_type, kDLCUDA) << "The device of v_a must be CUDA.";
  CHECK_EQ(s_a->device.device_type, kDLCUDA) << "The device of s_a must be CUDA.";
  CHECK_EQ(v_b->device.device_type, kDLCUDA) << "The device of v_b must be CUDA.";
  CHECK_EQ(s_b->device.device_type, kDLCUDA) << "The device of s_b must be CUDA.";
  CHECK_EQ(v_merged->device.device_type, kDLCUDA) << "The device of v_merged must be CUDA.";
  CHECK_EQ(s_merged->device.device_type, kDLCUDA) << "The device of s_merged must be CUDA.";
  int32_t dev_id = v_a->device.device_id;
  CHECK_EQ(s_a->device.device_id, dev_id);
  CHECK_EQ(v_b->device.device_id, dev_id);
  CHECK_EQ(s_b->device.device_id, dev_id);
  CHECK_EQ(v_merged->device.device_id, dev_id);
  CHECK_EQ(s_merged->device.device_id, dev_id);

  CHECK(v_a->dtype.lanes == 1 && s_a->dtype.lanes == 1 && v_b->dtype.lanes == 1 &&
        s_b->dtype.lanes == 1 && v_merged->dtype.lanes == 1 && s_merged->dtype.lanes == 1);
  CHECK(v_a->dtype.bits == v_b->dtype.bits && v_a->dtype.code == v_b->dtype.code);
  CHECK(s_a->dtype.bits == 32 && s_a->dtype.code == kDLFloat);
  CHECK(s_b->dtype.bits == 32 && s_b->dtype.code == kDLFloat);
  CHECK(s_merged->dtype.bits == 32 && s_merged->dtype.code == kDLFloat);

  CHECK_EQ(v_a->ndim, 3);
  int64_t batch_size = v_a->shape[0];
  int64_t num_heads = v_a->shape[1];
  int64_t head_dim = v_a->shape[2];
  CHECK_EQ(s_a->shape[0], batch_size);
  CHECK_EQ(s_a->shape[1], num_heads);
  CHECK_EQ(v_b->shape[0], batch_size);
  CHECK_EQ(v_b->shape[1], num_heads);
  CHECK_EQ(v_b->shape[2], head_dim);
  CHECK_EQ(s_b->shape[0], batch_size);
  CHECK_EQ(s_b->shape[1], num_heads);
  CHECK_EQ(v_merged->shape[0], batch_size);
  CHECK_EQ(v_merged->shape[1], num_heads);
  CHECK_EQ(v_merged->shape[2], head_dim);
  CHECK_EQ(s_merged->shape[0], batch_size);
  CHECK_EQ(s_merged->shape[1], num_heads);

  SWITCH_TVM_CUDA_DTYPE(v_a->dtype, dtype_in, {SWITCH_TVM_CUDA_DTYPE(v_merged->dtype, dtype_out, {
                          cudaError_t status = MergeState(
                              static_cast<dtype_in*>(v_a->data), static_cast<float*>(s_a->data),
                              static_cast<dtype_in*>(v_b->data), static_cast<float*>(s_b->data),
                              static_cast<dtype_out*>(v_merged->data),
                              static_cast<float*>(s_merged->data), batch_size, num_heads, head_dim);
                          if (status != cudaSuccess) {
                            LOG(FATAL)
                                << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
                          }
                        })});
}

void _FlashInferMergeStateInPlace(DLTensor* v, DLTensor* s, DLTensor* v_other, DLTensor* s_other) {
  CHECK_EQ(v->device.device_type, kDLCUDA) << "The device of v must be CUDA.";
  CHECK_EQ(s->device.device_type, kDLCUDA) << "The device of s must be CUDA.";
  CHECK_EQ(v_other->device.device_type, kDLCUDA) << "The device of v_other must be CUDA.";
  CHECK_EQ(s_other->device.device_type, kDLCUDA) << "The device of s_other must be CUDA.";
  int32_t dev_id = v->device.device_id;
  CHECK_EQ(s->device.device_id, dev_id);
  CHECK_EQ(v_other->device.device_id, dev_id);
  CHECK_EQ(s_other->device.device_id, dev_id);

  CHECK(v->dtype.lanes == 1 && s->dtype.lanes == 1 && v_other->dtype.lanes == 1 &&
        s_other->dtype.lanes == 1);
  CHECK(v->dtype.bits == v_other->dtype.bits && v->dtype.code == v_other->dtype.code);
  CHECK(s->dtype.bits == 32 && s->dtype.code == kDLFloat);
  CHECK(s_other->dtype.bits == 32 && s_other->dtype.code == kDLFloat);

  CHECK_EQ(v->ndim, 3);
  CHECK_EQ(v_other->ndim, 3);
  CHECK_EQ(s->ndim, 2);        // qo_nnz, nhead_qo
  CHECK_EQ(s_other->ndim, 2);  // qo_nnz, nhead_qo
  int64_t batch_size = v->shape[0];
  int64_t num_heads = v->shape[1];
  int64_t head_dim = v->shape[2];
  CHECK_EQ(s->shape[0], batch_size);
  CHECK_EQ(s->shape[1], num_heads);
  CHECK_EQ(v_other->shape[0], batch_size);
  CHECK_EQ(v_other->shape[1], num_heads);
  CHECK_EQ(v_other->shape[2], head_dim);
  CHECK_EQ(s_other->shape[0], batch_size);
  CHECK_EQ(s_other->shape[1], num_heads);

  SWITCH_TVM_CUDA_DTYPE(v->dtype, dtype, {
    cudaError_t status =
        MergeStateInPlace(static_cast<dtype*>(v->data), static_cast<float*>(s->data),
                          static_cast<dtype*>(v_other->data), static_cast<float*>(s_other->data),
                          batch_size, num_heads, head_dim);
    if (status != cudaSuccess) {
      LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
    }
  });
}

void _FlashInferBatchQKApplyRotaryInPlace(DLTensor* q, DLTensor* k, DLTensor* indptr,
                                          DLTensor* offsets, int64_t batch_size,
                                          int64_t num_qo_heads, int64_t num_kv_heads,
                                          int64_t head_dim, double rope_scale, double rope_theta) {
  SWITCH_TVM_CUDA_DTYPE(
      q->dtype, dtype, {SWITCH_TVM_CUDA_IDTYPE(indptr->dtype, idtype, {
        cudaError_t status = BatchQKApplyRotaryInPlace(
            static_cast<dtype*>(q->data), static_cast<dtype*>(k->data),
            static_cast<idtype*>(indptr->data), static_cast<idtype*>(offsets->data), batch_size,
            num_qo_heads, num_kv_heads, head_dim, rope_scale, rope_theta);
        if (status != cudaSuccess) {
          LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
        }
      })});
}

// TODO(Zihao): Unify the symbol names
TVM_REGISTER_GLOBAL("paged_kv_cache.attention_kernel_prefill")
    .set_body_typed(_FlashInferAttentionPrefillWithPagedKVCache);

TVM_REGISTER_GLOBAL("paged_kv_cache.attention_kernel_prefill_begin_forward")
    .set_body_typed(_FlashInferAttentionPrefillWithPagedKVCacheBeginForward);

TVM_REGISTER_GLOBAL("paged_kv_cache.attention_kernel_prefill_end_forward")
    .set_body_typed(_FlashInferAttentionPrefillWithPagedKVCacheEndForward);

TVM_REGISTER_GLOBAL("paged_kv_cache.attention_kernel_decode")
    .set_body_typed(_FlashInferAttentionDecodeWithPagedKVCache);

TVM_REGISTER_GLOBAL("paged_kv_cache.attention_kernel_decode_begin_forward")
    .set_body_typed(_FlashInferAttentionDecodeWithPagedKVCacheBeginForward);

TVM_REGISTER_GLOBAL("paged_kv_cache.attention_kernel_decode_end_forward")
    .set_body_typed(_FlashInferAttentionDecodeWithPagedKVCacheEndForward);

TVM_REGISTER_GLOBAL("flashinfer.attention_kernel_prefill_with_ragged_kv_cache")
    .set_body_typed(_FlashInferAttentionPrefillWithRaggedKVCache);

TVM_REGISTER_GLOBAL("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_begin_forward")
    .set_body_typed(_FlashInferAttentionPrefillWithRaggedKVCacheBeginForward);

TVM_REGISTER_GLOBAL("flashinfer.attention_kernel_prefill_with_ragged_kv_cache_end_forward")
    .set_body_typed(_FlashInferAttentionPrefillWithRaggedKVCacheEndForward);

TVM_REGISTER_GLOBAL("flashinfer.merge_state").set_body_typed(_FlashInferMergeState);

TVM_REGISTER_GLOBAL("flashinfer.merge_state_in_place").set_body_typed(_FlashInferMergeStateInPlace);

TVM_REGISTER_GLOBAL("flashinfer.batch_qk_apply_rotary_in_place")
    .set_body_typed(_FlashInferBatchQKApplyRotaryInPlace);

TVM_REGISTER_GLOBAL("flashinfer.single_prefill")
    .set_body_typed(_FlashInferSinglePrefillWithKVCache);
TVM_REGISTER_GLOBAL("flashinfer.single_decode").set_body_typed(_FlashInferSingleDecodeWithKVCache);
