/*
 * Copyright (c) 2026 by FlashInfer team.
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

#ifndef FLASHINFER_BATCH_MLA_PLAN_UPDATE_CUH_
#define FLASHINFER_BATCH_MLA_PLAN_UPDATE_CUH_

#include <cstdint>

#include "tvm_ffi_utils.h"

void CommitBatchMLACudaGraphPlanUpdate(
    TensorView live_int_workspace, TensorView live_qo_indptr, TensorView live_kv_indptr,
    TensorView live_kv_indices, TensorView live_kv_len_arr, TensorView candidate_int_workspace,
    TensorView candidate_qo_indptr, TensorView candidate_kv_indptr, TensorView source_kv_indices,
    TensorView candidate_kv_len_arr, int64_t staged_int_workspace_bytes,
    int64_t live_kv_indices_length);

#endif  // FLASHINFER_BATCH_MLA_PLAN_UPDATE_CUH_
