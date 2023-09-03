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

int _FlashInferSingleDecodeWithKVCache(DLTensor *q, DLTensor *k, DLTensor *v, DLTensor *tmp,
                                       int64_t rotary_mode, double rope_scale, double rope_theta,
                                       DLTensor *o) {
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
            (dtype_in *)q->data, (dtype_in *)k->data, (dtype_in *)v->data, (dtype_out *)o->data,
            (float *)tmp->data, num_heads, seq_len, head_dim, flashinfer::RotaryMode(rotary_mode),
            rope_scale, rope_theta, 0, dev_id);
        if (status != cudaSuccess) {
          LOG(FATAL) << "FlashInfer CUDA kernel error " << cudaGetErrorString(status);
        }
      })});
  return 0;
}

TVM_DLL_EXPORT_TYPED_FUNC(FlashInferSingleDecodeWithKVCache, _FlashInferSingleDecodeWithKVCache);