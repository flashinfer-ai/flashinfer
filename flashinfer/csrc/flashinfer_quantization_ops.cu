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
#include "pytorch_extension_utils.h"

void packbits(at::Tensor x, const std::string& bitorder, at::Tensor y);

void segment_packbits(at::Tensor x, at::Tensor input_indptr, at::Tensor output_indptr,
                      const std::string& bitorder, at::Tensor y);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // GPU packbits operator
  m.def("packbits", packbits);
  // GPU segment packbits operator
  m.def("segment_packbits", segment_packbits);
}
