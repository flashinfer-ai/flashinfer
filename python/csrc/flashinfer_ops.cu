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
#include <torch/extension.h>

#include "flashinfer_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("single_decode_with_kv_cache", &single_decode_with_kv_cache,
        "Single-request decode with KV-Cache operator");
  m.def("single_prefill_with_kv_cache", &single_prefill_with_kv_cache,
        "Single-request prefill with KV-Cache operator, return logsumexp");
  m.def("append_paged_kv_cache", &append_paged_kv_cache, "Append paged KV-Cache operator");
  m.def("merge_state", &merge_state, "Merge two self-attention states");
  m.def("merge_state_in_place", &merge_state_in_place,
        "Merge another self-attention state in-place.");
  m.def("merge_states", &merge_states, "Merge multiple self-attention states");
  m.def("batch_decode_with_padded_kv_cache", &batch_decode_with_padded_kv_cache,
        "Multi-request batch decode with padded KV-Cache operator");
  m.def("sampling_from_probs", &sampling_from_probs, "Sample from probabilities");
  m.def("top_k_sampling_from_probs", &top_k_sampling_from_probs,
        "Top-k sampling from probabilities");
  m.def("top_p_sampling_from_probs", &top_p_sampling_from_probs,
        "Top-p sampling from probabilities");
  m.def("rmsnorm", &rmsnorm, "Root mean square normalization");
  py::class_<BatchDecodeWithPagedKVCachePyTorchWrapper>(m,
                                                        "BatchDecodeWithPagedKVCachePyTorchWrapper")
      .def(py::init<unsigned int, unsigned int>())
      .def("begin_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::EndForward)
      .def("update_page_locked_buffer_size",
           &BatchDecodeWithPagedKVCachePyTorchWrapper::UpdatePageLockedBufferSize)
      .def("forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::Forward);
  py::class_<CUDAGraphBatchDecodeWithPagedKVCachePyTorchWrapper>(
      m, "CUDAGraphBatchDecodeWithPagedKVCachePyTorchWrapper")
      .def(py::init<unsigned int, unsigned int>())
      .def("begin_forward", &CUDAGraphBatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &CUDAGraphBatchDecodeWithPagedKVCachePyTorchWrapper::EndForward)
      .def("update_page_locked_buffer_size",
           &CUDAGraphBatchDecodeWithPagedKVCachePyTorchWrapper::UpdatePageLockedBufferSize)
      .def("forward", &CUDAGraphBatchDecodeWithPagedKVCachePyTorchWrapper::Forward);
  py::class_<BatchPrefillWithPagedKVCachePyTorchWrapper>(
      m, "BatchPrefillWithPagedKVCachePyTorchWrapper")
      .def(py::init<unsigned int, unsigned int>())
      .def("begin_forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::EndForward)
      .def("update_page_locked_buffer_size",
           &BatchPrefillWithPagedKVCachePyTorchWrapper::UpdatePageLockedBufferSize)
      .def("forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::Forward);
  py::class_<BatchPrefillWithRaggedKVCachePyTorchWrapper>(
      m, "BatchPrefillWithRaggedKVCachePyTorchWrapper")
      .def(py::init<unsigned int, unsigned int>())
      .def("begin_forward", &BatchPrefillWithRaggedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &BatchPrefillWithRaggedKVCachePyTorchWrapper::EndForward)
      .def("update_page_locked_buffer_size",
           &BatchPrefillWithRaggedKVCachePyTorchWrapper::UpdatePageLockedBufferSize)
      .def("forward", &BatchPrefillWithRaggedKVCachePyTorchWrapper::Forward);
}
