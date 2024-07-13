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

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

std::vector<torch::Tensor> merge_state(torch::Tensor v_a, torch::Tensor s_a, torch::Tensor v_b,
                                       torch::Tensor s_b) {
  CHECK_INPUT(v_a);
  CHECK_INPUT(s_a);
  CHECK_INPUT(v_b);
  CHECK_INPUT(s_b);
  auto device = v_a.device();
  CHECK_EQ(s_a.device(), device);
  CHECK_EQ(v_b.device(), device);
  CHECK_EQ(s_b.device(), device);
  CHECK_DIM(3, v_a);
  CHECK_DIM(2, s_a);
  CHECK_DIM(3, v_b);
  CHECK_DIM(2, s_b);
  CHECK_SHAPE(v_a, v_b);
  CHECK_SHAPE(s_a, s_b);
  CHECK_EQ(v_a.size(0), s_a.size(0));
  CHECK_EQ(v_a.size(1), s_b.size(1));
  s_a = s_a.to(torch::kFloat32);
  s_b = s_b.to(torch::kFloat32);
  unsigned int seq_len = v_a.size(0);
  unsigned int num_heads = v_a.size(1);
  unsigned int head_dim = v_a.size(2);
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto v_merged = torch::empty_like(v_a, v_a.options());
  auto s_merged = torch::empty({seq_len, num_heads}, s_a.options());

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(v_a.scalar_type(), c_type, [&] {
    cudaError_t status = MergeState(
        static_cast<c_type*>(v_a.data_ptr()), static_cast<float*>(s_a.data_ptr()),
        static_cast<c_type*>(v_b.data_ptr()), static_cast<float*>(s_b.data_ptr()),
        static_cast<c_type*>(v_merged.data_ptr()), static_cast<float*>(s_merged.data_ptr()),
        seq_len, num_heads, head_dim, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess,
                "MergeState kernel launch failed: ", cudaGetErrorString(status));
    return true;
  });

  TORCH_CHECK(success, "MergeState kernel launch failed: unsupported data type");
  return {v_merged, s_merged};
}

void merge_state_in_place(torch::Tensor v, torch::Tensor s, torch::Tensor v_other,
                          torch::Tensor s_other, std::optional<torch::Tensor> mask) {
  CHECK_INPUT(v);
  CHECK_INPUT(s);
  CHECK_INPUT(v_other);
  CHECK_INPUT(s_other);
  auto device = v.device();
  CHECK_EQ(s.device(), device);
  CHECK_EQ(v_other.device(), device);
  CHECK_EQ(s_other.device(), device);
  CHECK_DIM(3, v);
  CHECK_DIM(2, s);
  CHECK_DIM(3, v_other);
  CHECK_DIM(2, s_other);
  CHECK_SHAPE(v, v_other);
  CHECK_SHAPE(s, s_other);
  CHECK_EQ(v.size(0), s.size(0));
  CHECK_EQ(v.size(1), s.size(1));
  CHECK_EQ(s.scalar_type(), torch::kFloat32);
  CHECK_EQ(s_other.scalar_type(), torch::kFloat32);
  uint8_t* mask_ptr = nullptr;
  if (mask.has_value()) {
    CHECK_DIM(1, mask.value());
    CHECK_EQ(v.size(0), mask.value().size(0));
    CHECK_EQ(mask.value().device(), device);
    mask_ptr = static_cast<uint8_t*>(mask.value().data_ptr());
  }
  unsigned int seq_len = v.size(0);
  unsigned int num_heads = v.size(1);
  unsigned int head_dim = v.size(2);
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(v.scalar_type(), c_type, [&] {
    cudaError_t status = MergeStateInPlace(
        static_cast<c_type*>(v.data_ptr()), static_cast<float*>(s.data_ptr()),
        static_cast<c_type*>(v_other.data_ptr()), static_cast<float*>(s_other.data_ptr()), seq_len,
        num_heads, head_dim, mask_ptr, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess,
                "MergeStateInPlace kernel launch failed: ", cudaGetErrorString(status));
    return true;
  });

  TORCH_CHECK(success, "MergeStateInPlace kernel launch failed: unsupported data type");
}

std::vector<torch::Tensor> merge_states(torch::Tensor v, torch::Tensor s) {
  CHECK_INPUT(v);
  CHECK_INPUT(s);
  auto device = v.device();
  CHECK_EQ(s.device(), device);
  CHECK_DIM(4, v);
  CHECK_DIM(3, s);
  CHECK_EQ(v.size(0), s.size(0));
  CHECK_EQ(v.size(1), s.size(1));
  CHECK_EQ(v.size(2), s.size(2));
  unsigned int seq_len = v.size(0);
  unsigned int num_index_sets = v.size(1);
  unsigned int num_heads = v.size(2);
  unsigned int head_dim = v.size(3);
  s = s.to(torch::kFloat32);
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto v_merged = torch::empty({seq_len, num_heads, head_dim}, v.options());
  auto s_merged = torch::empty({seq_len, num_heads}, s.options());

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(v.scalar_type(), c_type, [&] {
    cudaError_t status = MergeStates(
        static_cast<c_type*>(v.data_ptr()), static_cast<float*>(s.data_ptr()),
        static_cast<c_type*>(v_merged.data_ptr()), static_cast<float*>(s_merged.data_ptr()),
        num_index_sets, seq_len, num_heads, head_dim, torch_current_stream);
    TORCH_CHECK(status == cudaSuccess,
                "MergeStates kernel launch failed: ", cudaGetErrorString(status));
    return true;
  });

  TORCH_CHECK(success, "MergeStates kernel launch failed: unsupported data type");
  return {v_merged, s_merged};
}
