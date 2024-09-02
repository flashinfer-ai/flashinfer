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
  m.def("bmm_fp8", &bmm_fp8, "BMM FP8");
  py::class_<CutlassSegmentGEMMPyTorchWrapper>(m, "CutlassSegmentGEMMPyTorchWrapper")
      .def(py::init<torch::Tensor>())
      .def("register_workspace", &CutlassSegmentGEMMPyTorchWrapper::RegisterWorkspaceBuffer)
      .def("run", &CutlassSegmentGEMMPyTorchWrapper::Run);
}
