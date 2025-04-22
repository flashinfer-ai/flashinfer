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
#include <flashinfer/attention/blackwell/fmha_cutlass_sm100.cuh>
#include <flashinfer/cutlass_utils.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

void FMHACutlassSM100Run(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor qo_indptr,
                         at::Tensor kv_indptr, at::Tensor o, std::optional<at::Tensor> maybe_lse,
                         int64_t mask_mode_code, double sm_scale, int64_t num_qo_heads,
                         int64_t num_kv_heads, int64_t head_dim, int64_t batch_size,
                         int64_t total_qo_len, int64_t total_kv_len, int64_t max_qo_len,
                         int64_t max_kv_len) {
  auto run = [&](auto shape) {
    if (mask_mode_code == 1) {
      FwdRunner<decltype(shape), void, CausalMask> runner;
      runner.run(q, k, v, qo_indptr, kv_indptr, o, maybe_lse, mask_mode_code, sm_scale,
                 num_qo_heads, num_kv_heads, head_dim, batch_size, total_qo_len, total_kv_len,
                 max_qo_len, max_kv_len);

    } else {
      FwdRunner<decltype(shape), void, NoMask> runner;
      runner.run(q, k, v, qo_indptr, kv_indptr, o, maybe_lse, mask_mode_code, sm_scale,
                 num_qo_heads, num_kv_heads, head_dim, batch_size, total_qo_len, total_kv_len,
                 max_qo_len, max_kv_len);
    }
  };
  run(Shape<_256, _128, _128>{});
}
