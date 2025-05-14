/*
 * Copyright (c) 2025 by FlashInfer team.
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

void CutlassGroupGemmGroupwiseScaledSM100(at::Tensor int_workspace_buffer,
                                          at::Tensor float_workspace_buffer, at::Tensor A,
                                          at::Tensor B, at::Tensor SFA, at::Tensor SFB,
                                          at::Tensor C, at::Tensor m_indptr, int64_t cum_m,
                                          int64_t n, int64_t k, int64_t scale_granularity_m,
                                          int64_t scale_granularity_n, int64_t scale_granularity_k);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("group_gemm_fp8_nt_groupwise", CutlassGroupGemmGroupwiseScaledSM100);
}
