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

void CutlassMLAPagedAttention(at::Tensor workspace, at::Tensor out, at::Tensor lse,
                              at::Tensor q_nope_pe, at::Tensor ckv_kpe_cache, at::Tensor kv_lens,
                              at::Tensor page_table);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // "Cutlass MLA Paged Attention"
  m.def("cutlass_mla_paged_attention", CutlassMLAPagedAttention);
}
