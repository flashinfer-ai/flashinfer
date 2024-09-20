"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

batch_prefill_paged_templ = r"""
#include <torch/extension.h>
#include <optional>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/handler.cuh>
#include "pytorch_extension_utils.h"

using namespace flashinfer;

using ParamsT = BatchPrefillPagedParams<{{ dtype_q }}, {{ dtype_kv }}, {{ dtype_o }}, {{ dtype_idx }}>;
using AttentionVariant = ComposedAttention<ParamsT, get_variant_code({{ use_custom_mask }}, {{ use_sliding_window }}, {{ use_logits_soft_cap }}, {{ use_alibi }})>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan", &BatchPrefillWithPagedKVCachePlan);
  m.def("run", &BatchPrefillWithPagedKVCacheRun);
}
"""