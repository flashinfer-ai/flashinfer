/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <flashinfer/attention/cutlass_mla.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;
using namespace flashinfer::attention;

void CutlassMLAPagedAttention(at::Tensor workspace, at::Tensor out, at::Tensor lse,
                              at::Tensor q_nope_pe, at::Tensor ckv_kpe_cache, at::Tensor kv_lens,
                              at::Tensor page_table) {
  const c10::cuda::OptionalCUDAGuard device_guard(q_nope_pe.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  int device_index = q_nope_pe.device().index();
  int batches = q_nope_pe.sizes()[0];
  int page_count_per_seq = page_table.sizes()[1];
  int page_count_total = ckv_kpe_cache.sizes()[0];
  int page_size = ckv_kpe_cache.sizes()[1];

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(q_nope_pe.scalar_type(), c_type, [&] {
    using cutlass_t = cutlass_dtype_t<c_type>;
    auto status = runMla<cutlass_t>(
        workspace.data_ptr(), out.data_ptr(), lse.data_ptr(), q_nope_pe.data_ptr(),
        ckv_kpe_cache.data_ptr(), kv_lens.data_ptr(), page_table.data_ptr(), batches,
        page_count_per_seq, page_count_total, page_size, device_index, stream);
    TORCH_CHECK(status == cudaSuccess,
                "Failed to run CutlassMLAPagedAttention: ", cudaGetErrorString(status));
    return true;
  });
}
