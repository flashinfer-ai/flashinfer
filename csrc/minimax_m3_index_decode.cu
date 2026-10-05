// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "tvm_ffi_utils.h"

namespace {
// Every visible block belongs to top-k when a valid host bound is <=2048.
// Runtime lengths still determine the visible prefix, including padded rows.
__global__ void index_decode_kernel(const int32_t* lengths, int32_t* output, int64_t stride) {
  int b = blockIdx.x;
  int slot = threadIdx.x;
  int visible = (lengths[b] + 127) >> 7;
  if (slot < 16) output[b * stride + slot] = slot < visible ? slot : -1;
}
}  // namespace

void minimax_m3_index_decode(TensorView q, TensorView cache, TensorView table, TensorView lengths,
                             TensorView out, int64_t max_seq_len) {
  CHECK_CUDA(q);
  CHECK_CUDA(cache);
  CHECK_CUDA(table);
  CHECK_INPUT(lengths);
  CHECK_CUDA(out);
  CHECK_DEVICE(cache, q);
  CHECK_DEVICE(table, q);
  CHECK_DEVICE(lengths, q);
  CHECK_DEVICE(out, q);
  CHECK_INPUT_TYPE(q, dl_bfloat16);
  CHECK_INPUT_TYPE(cache, dl_bfloat16);
  CHECK_INPUT_TYPE(table, dl_int32);
  CHECK_INPUT_TYPE(lengths, dl_int32);
  CHECK_INPUT_TYPE(out, dl_int32);
  CHECK_DIM(3, q);
  CHECK_DIM(3, cache);
  CHECK_DIM(2, table);
  CHECK_DIM(1, lengths);
  CHECK_DIM(3, out);
  auto b = q.size(0);
  TVM_FFI_ICHECK(b == 8 || b == 16);
  TVM_FFI_ICHECK_EQ(q.size(1), 1);
  TVM_FFI_ICHECK_EQ(q.size(2), 128);
  TVM_FFI_ICHECK_EQ(q.stride(2), 1);
  TVM_FFI_ICHECK_EQ(cache.size(1), 128);
  TVM_FFI_ICHECK_EQ(cache.size(2), 128);
  TVM_FFI_ICHECK_EQ(cache.stride(2), 1);
  TVM_FFI_ICHECK_EQ(table.size(0), b);
  TVM_FFI_ICHECK_EQ(table.size(1), 128);
  TVM_FFI_ICHECK_EQ(table.stride(1), 1);
  TVM_FFI_ICHECK_EQ(lengths.size(0), b);
  TVM_FFI_ICHECK_EQ(out.size(0), 1);
  TVM_FFI_ICHECK_EQ(out.size(1), b);
  TVM_FFI_ICHECK_EQ(out.size(2), 16);
  TVM_FFI_ICHECK_EQ(out.stride(2), 1);
  TVM_FFI_ICHECK(max_seq_len > 0 && max_seq_len <= 2048);
  ffi::CUDADeviceGuard guard(q.device().device_id);
  index_decode_kernel<<<b, 32, 0, get_stream(q.device())>>>(
      static_cast<const int32_t*>(lengths.data_ptr()), static_cast<int32_t*>(out.data_ptr()),
      out.stride(1));
  auto status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_m3_index_decode, minimax_m3_index_decode);
