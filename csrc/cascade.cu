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
#include <flashinfer/attention/cascade.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Optional;

void merge_state(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor v_merged, Tensor s_merged) {
  CHECK_INPUT(v_a);
  CHECK_INPUT(s_a);
  CHECK_INPUT(v_b);
  CHECK_INPUT(s_b);
  CHECK_DEVICE(s_a, v_a);
  CHECK_DEVICE(v_b, v_a);
  CHECK_DEVICE(s_b, v_a);
  CHECK_DIM(3, v_a);
  CHECK_DIM(2, s_a);
  CHECK_DIM(3, v_b);
  CHECK_DIM(2, s_b);
  CHECK_SHAPE(v_a, v_b);
  CHECK_SHAPE(s_a, s_b);
  TVM_FFI_ICHECK_EQ(v_a->shape[0], s_a->shape[0]);
  TVM_FFI_ICHECK_EQ(v_a->shape[1], s_b->shape[1]);
  unsigned int seq_len = v_a->shape[0];
  unsigned int num_heads = v_a->shape[1];
  unsigned int head_dim = v_a->shape[2];

  cudaSetDevice(v_a->device.device_id);
  auto stream = get_stream(v_a->device);

  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(v_a->dtype, c_type, [&] {
    cudaError_t status =
        MergeState(static_cast<c_type*>(v_a->data), static_cast<float*>(s_a->data),
                   static_cast<c_type*>(v_b->data), static_cast<float*>(s_b->data),
                   static_cast<c_type*>(v_merged->data), static_cast<float*>(s_merged->data),
                   seq_len, num_heads, head_dim, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "MergeState kernel launch failed: " << cudaGetErrorString(status);
    return true;
  });
  TVM_FFI_ICHECK(success) << "MergeState kernel launch failed: unsupported data type.";
}

void merge_state_in_place(Tensor v, Tensor s, Tensor v_other, Tensor s_other,
                          Optional<Tensor> mask) {
  CHECK_INPUT(v);
  CHECK_INPUT(s);
  CHECK_INPUT(v_other);
  CHECK_INPUT(s_other);
  CHECK_DEVICE(s, v);
  CHECK_DEVICE(v_other, v);
  CHECK_DEVICE(s_other, v);
  CHECK_DIM(3, v);
  CHECK_DIM(2, s);
  CHECK_DIM(3, v_other);
  CHECK_DIM(2, s_other);
  CHECK_SHAPE(v, v_other);
  CHECK_SHAPE(s, s_other);
  TVM_FFI_ICHECK_EQ(v->shape[0], s->shape[0]);
  TVM_FFI_ICHECK_EQ(v->shape[1], s->shape[1]);
  uint8_t* mask_ptr = nullptr;
  if (mask.has_value()) {
    CHECK_DIM(1, mask.value());
    TVM_FFI_ICHECK_EQ(v->shape[0], mask.value()->shape[0]);
    CHECK_DEVICE(mask.value(), v);
    mask_ptr = static_cast<uint8_t*>(mask.value()->data);
  }
  unsigned int seq_len = v->shape[0];
  unsigned int num_heads = v->shape[1];
  unsigned int head_dim = v->shape[2];

  cudaSetDevice(v->device.device_id);
  auto stream = get_stream(v->device);
  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(v->dtype, c_type, [&] {
    cudaError_t status =
        MergeStateInPlace(static_cast<c_type*>(v->data), static_cast<float*>(s->data),
                          static_cast<c_type*>(v_other->data), static_cast<float*>(s_other->data),
                          seq_len, num_heads, head_dim, mask_ptr, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "MergeStateInPlace kernel launch failed: " << cudaGetErrorString(status);
    return true;
  });

  TVM_FFI_ICHECK(success) << "MergeStateInPlace kernel launch failed: unsupported data type.";
}

void merge_states(Tensor v, Tensor s, Tensor v_merged, Tensor s_merged) {
  CHECK_INPUT(v);
  CHECK_INPUT(s);
  CHECK_DEVICE(s, v);
  CHECK_DIM(4, v);
  CHECK_DIM(3, s);
  TVM_FFI_ICHECK_EQ(v->shape[0], s->shape[0]);
  TVM_FFI_ICHECK_EQ(v->shape[1], s->shape[1]);
  TVM_FFI_ICHECK_EQ(v->shape[2], s->shape[2]);
  unsigned int seq_len = v->shape[0];
  unsigned int num_index_sets = v->shape[1];
  unsigned int num_heads = v->shape[2];
  unsigned int head_dim = v->shape[3];

  cudaSetDevice(v->device.device_id);
  auto stream = get_stream(v->device);
  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(v->dtype, c_type, [&] {
    cudaError_t status =
        MergeStates(static_cast<c_type*>(v->data), static_cast<float*>(s->data),
                    static_cast<c_type*>(v_merged->data), static_cast<float*>(s_merged->data),
                    num_index_sets, seq_len, num_heads, head_dim, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "MergeStates kernel launch failed: " << cudaGetErrorString(status);
    return true;
  });

  TVM_FFI_ICHECK(success) << "MergeStates kernel launch failed: unsupported data type.";
}
