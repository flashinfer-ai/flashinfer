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

#define SWITCH_TVM_CUDA_DTYPE(dl_dtype, cuda_dtype, ...)          \
  if (dl_dtype.code == kDLFloat && dl_dtype.bits == 16) {         \
    using cuda_dtype = half;                                      \
    __VA_ARGS__                                                   \
  } else if (dl_dtype.code == kDLFloat && dl_dtype.bits == 32) {  \
    using cuda_dtype = float;                                     \
    __VA_ARGS__                                                   \
  } else if (dl_dtype.code == kDLBfloat && dl_dtype.bits == 16) { \
    using cuda_dtype = nv_bfloat16;                               \
    __VA_ARGS__                                                   \
  } else if (dl_dtype.code == DataType::kE4M3Float) {             \
    using cuda_dtype = __nv_fp8_e4m3;                             \
    __VA_ARGS__                                                   \
  } else if (dl_dtype.code == DataType::kE5M2Float) {             \
    using cuda_dtype = __nv_fp8_e5m2;                             \
    __VA_ARGS__                                                   \
  } else {                                                        \
    LOG(FATAL) << "Unsupported data type " << dl_dtype.code;      \
  }

#define SWITCH_TVM_CUDA_IDTYPE(dl_dtype, cuda_dtype, ...)      \
  if (dl_dtype.code == kDLInt && dl_dtype.bits == 32) {        \
    using cuda_dtype = int32_t;                                \
    __VA_ARGS__                                                \
  } else if (dl_dtype.code == kDLInt && dl_dtype.bits == 64) { \
    using cuda_dtype = int64_t;                                \
    __VA_ARGS__                                                \
  } else {                                                     \
    LOG(FATAL) << "Unsupported data type " << dl_dtype.code;   \
  }

int _FlashInferSingleDecodeWithKVCache(DLTensor* q, DLTensor* k, DLTensor* v, DLTensor* tmp,
                                       int64_t qkv_layout, int64_t rotary_mode, double rope_scale,
                                       double rope_theta, DLTensor* o) {
  CHECK_EQ(q->device.device_type, kDLCUDA) << "The device of q matrix must be CUDA.";
  CHECK_EQ(k->device.device_type, kDLCUDA) << "The device of k matrix must be CUDA.";
  CHECK_EQ(v->device.device_type, kDLCUDA) << "The device of v matrix must be CUDA.";
  CHECK_EQ(tmp->device.device_type, kDLCUDA) << "The device of tmp matrix must be CUDA.";
  CHECK_EQ(o->device.device_type, kDLCUDA) << "The device of o matrix must be CUDA.";

  size_t dev_id = q->device.device_id;
  CHECK_EQ(k->device.device_id, dev_id) << "The device id of q and k matrix doesn't match.";
  CHECK_EQ(v->device.device_id, dev_id) << "The device id of q and v matrix doesn't match.";
  CHECK_EQ(tmp->device.device_id, dev_id) << "The device id of q and tmp matrix doesn't match.";
  CHECK_EQ(o->device.device_id, dev_id) << "The device id of q and o matrix doesn't match.";

  CHECK_EQ(q->ndim, 2);
  size_t num_heads = q->shape[0];
  size_t head_dim = q->shape[1];
  CHECK_EQ(k->ndim, 3);
  size_t seq_len = k->shape[0];
  CHECK_EQ(k->shape[1], num_heads);
  CHECK_EQ(k->shape[2], head_dim);
  CHECK_EQ(v->ndim, 3);
  CHECK_EQ(v->shape[0], seq_len);
  CHECK_EQ(v->shape[1], num_heads);
  CHECK_EQ(v->shape[2], head_dim);
  CHECK_EQ(o->ndim, 2);
  CHECK_EQ(o->shape[0], num_heads);
  CHECK_EQ(o->shape[1], head_dim);

  CHECK(q->dtype.lanes == 1 && k->dtype.lanes == 1 && v->dtype.lanes == 1);
  CHECK(q->dtype.bits == k->dtype.bits && q->dtype.code == k->dtype.code);
  CHECK(q->dtype.bits == v->dtype.bits && q->dtype.code == v->dtype.code);

  SWITCH_TVM_CUDA_DTYPE(
      q->dtype, dtype_in, {SWITCH_TVM_CUDA_DTYPE(o->dtype, dtype_out, {
        cudaError_t status = flashinfer::SingleDecodeWithKVCache(
            (dtype_in*)q->data, (dtype_in*)k->data, (dtype_in*)v->data, (dtype_out*)o->data,
            (float*)tmp->data, num_heads, seq_len, head_dim, flashinfer::QKVLayout(qkv_layout),
            flashinfer::RotaryMode(rotary_mode), rope_scale, rope_theta, 0, dev_id);
        if (status != cudaSuccess) {
          LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
        }
      })});
  return 0;
}

TVM_DLL_EXPORT_TYPED_FUNC(FlashInferSingleDecodeWithKVCache, _FlashInferSingleDecodeWithKVCache);

void _FlashInferBatchDecodeWithPagedKVCache(DLTensor* q_data, DLTensor* pages,
                                            DLTensor* page_table_indptr,
                                            DLTensor* page_table_values,
                                            DLTensor* last_page_offset,  //
                                            int64_t layer_id,            //
                                            DLTensor* output) {
  CHECK_EQ(q_data->device.device_type, kDLCUDA) << "The device of q_data must be CUDA.";
  CHECK_EQ(pages->device.device_type, kDLCUDA) << "The device of kv pages must be CUDA.";
  CHECK_EQ(page_table_indptr->device.device_type, kDLCUDA)
      << "The device of page_table_indptr matrix must be CUDA.";
  CHECK_EQ(page_table_values->device.device_type, kDLCUDA)
      << "The device of page_table_values matrix must be CUDA.";
  CHECK_EQ(last_page_offset->device.device_type, kDLCUDA)
      << "The device of last_page_offset matrix must be CUDA.";
  CHECK_EQ(output->device.device_type, kDLCUDA) << "The device of output must be CUDA.";

  int32_t dev_id = q_data->device.device_id;
  CHECK_EQ(pages->device.device_id, dev_id);
  CHECK_EQ(page_table_indptr->device.device_id, dev_id);
  CHECK_EQ(page_table_values->device.device_id, dev_id);
  CHECK_EQ(last_page_offset->device.device_id, dev_id);
  CHECK_EQ(output->device.device_id, dev_id);

  CHECK(q_data->dtype.lanes == 1 && pages->dtype.lanes == 1 && output->dtype.lanes == 1);
  CHECK(q_data->dtype.bits == pages->dtype.bits && q_data->dtype.code == pages->dtype.code);
  CHECK(page_table_indptr->dtype.lanes == 1 && page_table_values->dtype.lanes == 1 &&
        last_page_offset->dtype.lanes == 1);
  CHECK(page_table_indptr->dtype.bits == page_table_values->dtype.bits &&
        page_table_indptr->dtype.bits == last_page_offset->dtype.bits &&
        page_table_indptr->dtype.code == page_table_values->dtype.code &&
        page_table_indptr->dtype.code == last_page_offset->dtype.code);

  CHECK_EQ(pages->ndim, 7);
  CHECK_LT(layer_id, pages->shape[1]);
  CHECK_GE(layer_id, 0);
  CHECK_EQ(pages->shape[2], 1) << "Page chunk size should be fixed to 1 right now.";
  CHECK_EQ(pages->shape[3], 2);
  int64_t npage = pages->shape[0];
  int64_t nlayer = pages->shape[1];
  int64_t nhead = pages->shape[4];
  int64_t nfeat = pages->shape[6];
  int64_t page_size = pages->shape[5];

  CHECK_EQ(last_page_offset->ndim, 1);
  int64_t num_total_seqs = last_page_offset->shape[0];

  CHECK_EQ(page_table_indptr->ndim, 1);
  CHECK_EQ(page_table_indptr->shape[0], num_total_seqs + 1);
  CHECK_EQ(page_table_values->ndim, 1);

  CHECK_EQ(q_data->ndim, 4);
  CHECK_EQ(q_data->shape[0], num_total_seqs);
  CHECK_EQ(q_data->shape[1], 1);
  CHECK_EQ(q_data->shape[2], nhead);
  CHECK_EQ(q_data->shape[3], nfeat);

  CHECK_EQ(output->ndim, 4);
  CHECK_EQ(output->shape[0], num_total_seqs);
  CHECK_EQ(output->shape[1], 1);
  CHECK_EQ(output->shape[2], nhead);
  CHECK_EQ(output->shape[3], nfeat);

  SWITCH_TVM_CUDA_DTYPE(
      pages->dtype, dtype_in,
      {SWITCH_TVM_CUDA_DTYPE(
          output->dtype, dtype_out, {SWITCH_TVM_CUDA_IDTYPE(page_table_values->dtype, dtype_idx, {
            flashinfer::paged_kv_t<dtype_in, dtype_idx> cache(
                npage, nlayer, layer_id, nhead, page_size, nfeat, num_total_seqs,
                static_cast<dtype_in*>(pages->data),
                static_cast<dtype_idx*>(page_table_indptr->data),
                static_cast<dtype_idx*>(page_table_values->data),
                static_cast<dtype_idx*>(last_page_offset->data));
            cudaError_t status = flashinfer::BatchDecodeWithPagedKVCache<dtype_in, dtype_out>(
                (dtype_in*)q_data->data, cache, static_cast<dtype_out*>(output->data), nullptr,
                flashinfer::RotaryMode::kNone, 1.0f, 1e4, 0, q_data->device.device_id);
            if (status != cudaSuccess) {
              LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
            }
          })})});
}

TVM_DLL_EXPORT_TYPED_FUNC(FlashInferBatchDecodeWithPagedKVCache,
                          _FlashInferBatchDecodeWithPagedKVCache);
