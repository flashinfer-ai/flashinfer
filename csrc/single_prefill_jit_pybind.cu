/*
 * Copyright (c) 2023-2025 by FlashInfer team.
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
#include "single_prefill_config.inc"

void single_prefill_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                  at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                  unsigned int mask_mode_code, unsigned int layout,
                                  int32_t window_left ADDITIONAL_FUNC_PARAMS, int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &single_prefill_with_kv_cache,
        "Single-request prefill attention with KV-Cache operator");
}
