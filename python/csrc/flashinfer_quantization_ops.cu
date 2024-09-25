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

torch::Tensor packbits(torch::Tensor x, const std::string& bitorder);

torch::Tensor segment_packbits(torch::Tensor x, torch::Tensor input_indptr,
                               torch::Tensor output_indptr, const std::string& bitorder);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("packbits", &packbits, "GPU packbits operator");
  m.def("segment_packbits", &segment_packbits, "GPU segment packbits operator");
}
