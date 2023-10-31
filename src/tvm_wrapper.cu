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

using tvm::runtime::DataType;
using tvm::runtime::NDArray;
using tvm::runtime::ShapeTuple;

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

int _FlashInferSingleDecodeWithKVCache(DLTensor* q, DLTensor* k, DLTensor* v,
                                       DLTensor* tmp, int64_t qkv_layout,
                                       int64_t rotary_mode, double rope_scale,
                                       double rope_theta, DLTensor* o) {
  CHECK_EQ(q->device.device_type, kDLCUDA)
      << "The device of q matrix must be CUDA.";
  CHECK_EQ(k->device.device_type, kDLCUDA)
      << "The device of k matrix must be CUDA.";
  CHECK_EQ(v->device.device_type, kDLCUDA)
      << "The device of v matrix must be CUDA.";
  CHECK_EQ(tmp->device.device_type, kDLCUDA)
      << "The device of tmp matrix must be CUDA.";
  CHECK_EQ(o->device.device_type, kDLCUDA)
      << "The device of o matrix must be CUDA.";

  size_t dev_id = q->device.device_id;
  CHECK_EQ(k->device.device_id, dev_id)
      << "The device id of q and k matrix doesn't match.";
  CHECK_EQ(v->device.device_id, dev_id)
      << "The device id of q and v matrix doesn't match.";
  CHECK_EQ(tmp->device.device_id, dev_id)
      << "The device id of q and tmp matrix doesn't match.";
  CHECK_EQ(o->device.device_id, dev_id)
      << "The device id of q and o matrix doesn't match.";

  CHECK_EQ(q->ndim, 2);
  size_t num_qo_heads = q->shape[0];
  size_t head_dim = q->shape[1];
  CHECK_EQ(k->ndim, 3);
  size_t num_kv_heads = k->shape[1];
  size_t seq_len = k->shape[0];
  CHECK_EQ(k->shape[2], head_dim);
  CHECK_EQ(v->ndim, 3);
  CHECK_EQ(v->shape[0], seq_len);
  CHECK_EQ(v->shape[1], num_kv_heads);
  CHECK_EQ(v->shape[2], head_dim);
  CHECK_EQ(o->ndim, 2);
  CHECK_EQ(o->shape[0], num_qo_heads);
  CHECK_EQ(o->shape[1], head_dim);

  CHECK(q->dtype.lanes == 1 && k->dtype.lanes == 1 && v->dtype.lanes == 1);
  CHECK(q->dtype.bits == k->dtype.bits && q->dtype.code == k->dtype.code);
  CHECK(q->dtype.bits == v->dtype.bits && q->dtype.code == v->dtype.code);

  SWITCH_TVM_CUDA_DTYPE(
      q->dtype, dtype_in, {SWITCH_TVM_CUDA_DTYPE(o->dtype, dtype_out, {
        cudaError_t status = flashinfer::SingleDecodeWithKVCache(
            (dtype_in*)q->data, (dtype_in*)k->data, (dtype_in*)v->data,
            (dtype_out*)o->data, (float*)tmp->data, num_qo_heads, num_kv_heads,
            seq_len, head_dim, flashinfer::QKVLayout(qkv_layout),
            flashinfer::RotaryMode(rotary_mode), rope_scale, rope_theta, 0);
        if (status != cudaSuccess) {
          LOG(FATAL) << "FlashInfer CUDA kernel error "
                     << cudaGetErrorString(status);
        }
      })});
  return 0;
}

TVM_DLL_EXPORT_TYPED_FUNC(FlashInferSingleDecodeWithKVCache,
                          _FlashInferSingleDecodeWithKVCache);

void _FlashInferAttentionWithPagedKVCache(DLTensor* q_data, DLTensor* pages,
                                          DLTensor* page_table_indptr,
                                          DLTensor* page_table_values,
                                          DLTensor* last_page_offset,      //
                                          DLTensor* append_length_indptr,  //
                                          int64_t layer_id,                //
                                          DLTensor* tmp_buffer,            //
                                          DLTensor* output,                //
                                          int64_t rotary_mode = 0,         //
                                          double rope_scale = 1.0f,        //
                                          double rope_theta = 1e4) {
  CHECK_EQ(q_data->device.device_type, kDLCUDA)
      << "The device of q_data must be CUDA.";
  CHECK_EQ(pages->device.device_type, kDLCUDA)
      << "The device of kv pages must be CUDA.";
  CHECK_EQ(page_table_indptr->device.device_type, kDLCUDA)
      << "The device of page_table_indptr matrix must be CUDA.";
  CHECK_EQ(page_table_values->device.device_type, kDLCUDA)
      << "The device of page_table_values matrix must be CUDA.";
  CHECK_EQ(last_page_offset->device.device_type, kDLCUDA)
      << "The device of last_page_offset matrix must be CUDA.";
  CHECK_EQ(append_length_indptr->device.device_type, kDLCUDA)
      << "The device of append_length_indptr matrix must be CUDA.";
  CHECK_EQ(tmp_buffer->device.device_type, kDLCUDA)
      << "The device of the tmp buffer must be CUDA.";
  CHECK_EQ(output->device.device_type, kDLCUDA)
      << "The device of output must be CUDA.";

  int32_t dev_id = q_data->device.device_id;
  CHECK_EQ(pages->device.device_id, dev_id);
  CHECK_EQ(page_table_indptr->device.device_id, dev_id);
  CHECK_EQ(page_table_values->device.device_id, dev_id);
  CHECK_EQ(last_page_offset->device.device_id, dev_id);
  CHECK_EQ(append_length_indptr->device.device_id, dev_id);
  CHECK_EQ(tmp_buffer->device.device_id, dev_id);
  CHECK_EQ(output->device.device_id, dev_id);

  CHECK(q_data->dtype.lanes == 1 && pages->dtype.lanes == 1 &&
        output->dtype.lanes == 1);
  CHECK(q_data->dtype.bits == pages->dtype.bits &&
        q_data->dtype.code == pages->dtype.code);
  CHECK(page_table_indptr->dtype.lanes == 1 &&
        page_table_values->dtype.lanes == 1 &&
        last_page_offset->dtype.lanes == 1 &&
        append_length_indptr->dtype.lanes == 1);
  CHECK(page_table_indptr->dtype.bits == page_table_values->dtype.bits &&
        page_table_indptr->dtype.bits == last_page_offset->dtype.bits &&
        page_table_indptr->dtype.bits == append_length_indptr->dtype.bits &&
        page_table_indptr->dtype.code == page_table_values->dtype.code &&
        page_table_indptr->dtype.code == last_page_offset->dtype.code &&
        page_table_indptr->dtype.code == append_length_indptr->dtype.code);

  CHECK_EQ(pages->ndim, 6);
  CHECK_LT(layer_id, pages->shape[1]);
  CHECK_GE(layer_id, 0);
  CHECK_EQ(pages->shape[2], 2);
  int64_t nlayer = pages->shape[1];
  int64_t nhead_kv = pages->shape[3];
  int64_t nhead_qo = q_data->shape[2];
  int64_t nfeat = pages->shape[5];
  int64_t page_size = pages->shape[4];

  CHECK_EQ(last_page_offset->ndim, 1);
  int64_t num_total_seqs = last_page_offset->shape[0];

  CHECK_EQ(append_length_indptr->ndim, 1);
  CHECK_EQ(append_length_indptr->shape[0], num_total_seqs + 1);
  int total_append_length;

  std::vector<int> append_lengths;
  SWITCH_TVM_CUDA_IDTYPE(append_length_indptr->dtype, dtype_idx, {
    std::vector<dtype_idx> append_length_indptr_vec;
    append_length_indptr_vec.resize(num_total_seqs + 1);
    cudaMemcpy(append_length_indptr_vec.data(),
               static_cast<dtype_idx*>(append_length_indptr->data),
               (num_total_seqs + 1) * sizeof(dtype_idx),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_total_seqs; ++i) {
      CHECK_LE(append_length_indptr_vec[i], append_length_indptr_vec[i + 1]);
      append_lengths.push_back(append_length_indptr_vec[i + 1] -
                               append_length_indptr_vec[i]);
    }
    total_append_length =
        static_cast<int>(append_length_indptr_vec[num_total_seqs]);
  });

  CHECK_EQ(page_table_indptr->ndim, 1);
  CHECK_EQ(page_table_indptr->shape[0], num_total_seqs + 1);
  CHECK_EQ(page_table_values->ndim, 1);

  CHECK_EQ(q_data->ndim, 4);
  CHECK_EQ(output->ndim, 4);
  CHECK_EQ(q_data->shape[0], output->shape[0]);
  CHECK_EQ(q_data->shape[1], output->shape[1]);
  CHECK_EQ(q_data->shape[3], nfeat);
  CHECK_EQ(output->shape[2], nhead_qo);
  CHECK_EQ(output->shape[3], nfeat);

  // - Detect the decode pattern: all append lenghts are 1.
  if (std::all_of(append_lengths.begin(), append_lengths.end(),
                  [](int length) { return length == 1; }) &&
      q_data->shape[0] == num_total_seqs && q_data->shape[1] == 1) {
    SWITCH_TVM_CUDA_DTYPE(
        pages->dtype, dtype_in,
        {SWITCH_TVM_CUDA_DTYPE(
            output->dtype, dtype_out,
            {SWITCH_TVM_CUDA_IDTYPE(page_table_values->dtype, dtype_idx, {
              flashinfer::paged_kv_t<dtype_in, dtype_idx> cache(
                  nlayer, layer_id, nhead_kv, page_size, nfeat, num_total_seqs,
                  static_cast<dtype_in*>(pages->data),
                  static_cast<dtype_idx*>(page_table_indptr->data),
                  static_cast<dtype_idx*>(page_table_values->data),
                  static_cast<dtype_idx*>(last_page_offset->data));
              cudaError_t status =
                  flashinfer::BatchDecodeWithPagedKVCache<dtype_in, dtype_out>(
                      static_cast<dtype_in*>(q_data->data), cache,
                      static_cast<dtype_out*>(output->data),
                      nullptr, nhead_qo,
                      flashinfer::RotaryMode(rotary_mode), rope_scale,
                      rope_theta, 0);
              if (status != cudaSuccess) {
                LOG(FATAL) << "FlashInfer CUDA kernel error "
                           << cudaGetErrorString(status);
              }
            })})});
  } else {
    CHECK_EQ(q_data->shape[0], 1);
    CHECK_EQ(q_data->shape[1], total_append_length);
    for (int length : append_lengths) {
      CHECK(length == 0 || length == total_append_length)
          << "Only single-sequence prefill is supported for now.";
    }
    SWITCH_TVM_CUDA_DTYPE(
        pages->dtype, dtype_in,
        {SWITCH_TVM_CUDA_DTYPE(
            output->dtype, dtype_out,
            {SWITCH_TVM_CUDA_IDTYPE(page_table_values->dtype, dtype_idx, {
              flashinfer::paged_kv_t<dtype_in, dtype_idx> cache(
                  nlayer, layer_id, nhead_kv, page_size, nfeat, num_total_seqs,
                  static_cast<dtype_in*>(pages->data),
                  static_cast<dtype_idx*>(page_table_indptr->data),
                  static_cast<dtype_idx*>(page_table_values->data),
                  static_cast<dtype_idx*>(last_page_offset->data));
              cudaError_t status =
                  flashinfer::BatchPrefillWithPagedKVCache<dtype_in, dtype_out,
                                                           dtype_idx>(
                      static_cast<dtype_in*>(q_data->data), cache,
                      static_cast<dtype_idx*>(append_length_indptr->data),
                      static_cast<dtype_out*>(output->data),
                      nullptr, nhead_qo,
                      /*causal=*/true, flashinfer::RotaryMode(rotary_mode),
                      rope_scale, rope_theta, 0);
              if (status != cudaSuccess) {
                LOG(FATAL) << "FlashInfer CUDA kernel error "
                           << cudaGetErrorString(status);
              }
            })})});
  }
}

TVM_DLL_EXPORT_TYPED_FUNC(FlashInferAttentionWithPagedKVCache,
                          _FlashInferAttentionWithPagedKVCache);
