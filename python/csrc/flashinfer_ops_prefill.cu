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

#include "flashinfer_ops_prefill.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("single_prefill_with_kv_cache", &single_prefill_with_kv_cache,
        "Single-request prefill with KV-Cache operator, return logsumexp");
  m.def(
      "single_prefill_with_kv_cache_custom_mask", &single_prefill_with_kv_cache_custom_mask,
      "Single-request prefill with KV-Cache operator, user defined custom mask, return logsumexp");
  py::class_<BatchPrefillWithPagedKVCachePyTorchWrapper>(
      m, "BatchPrefillWithPagedKVCachePyTorchWrapper")
      .def(py::init<unsigned int, bool>())
      .def("begin_forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::EndForward)
      .def("is_cuda_graph_enabled", &BatchPrefillWithPagedKVCachePyTorchWrapper::IsCUDAGraphEnabled)
      .def("update_page_locked_buffer_size",
           &BatchPrefillWithPagedKVCachePyTorchWrapper::UpdatePageLockedBufferSize)
      .def("forward", &BatchPrefillWithPagedKVCachePyTorchWrapper::Forward)
      .def("forward_custom_mask", &BatchPrefillWithPagedKVCachePyTorchWrapper::ForwardCustomMask);
  py::class_<BatchPrefillWithRaggedKVCachePyTorchWrapper>(
      m, "BatchPrefillWithRaggedKVCachePyTorchWrapper")
      .def(py::init<unsigned int, bool>())
      .def("begin_forward", &BatchPrefillWithRaggedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &BatchPrefillWithRaggedKVCachePyTorchWrapper::EndForward)
      .def("is_cuda_graph_enabled",
           &BatchPrefillWithRaggedKVCachePyTorchWrapper::IsCUDAGraphEnabled)
      .def("update_page_locked_buffer_size",
           &BatchPrefillWithRaggedKVCachePyTorchWrapper::UpdatePageLockedBufferSize)
      .def("forward", &BatchPrefillWithRaggedKVCachePyTorchWrapper::Forward)
      .def("forward_custom_mask", &BatchPrefillWithRaggedKVCachePyTorchWrapper::ForwardCustomMask);
}
