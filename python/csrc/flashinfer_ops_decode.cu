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

#include "flashinfer_ops_decode.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("single_decode_with_kv_cache", &single_decode_with_kv_cache,
        "Single-request decode with KV-Cache operator");
  py::class_<BatchDecodeWithPagedKVCachePyTorchWrapper>(m,
                                                        "BatchDecodeWithPagedKVCachePyTorchWrapper")
      .def(py::init<unsigned int, bool, unsigned int>())
      .def("begin_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::EndForward)
      .def("is_cuda_graph_enabled", &BatchDecodeWithPagedKVCachePyTorchWrapper::IsCUDAGraphEnabled)
      .def("update_page_locked_buffer_size",
           &BatchDecodeWithPagedKVCachePyTorchWrapper::UpdatePageLockedBufferSize)
      .def("forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::Forward);
}
