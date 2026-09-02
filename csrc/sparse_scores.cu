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
#include <flashinfer/attention/sparse_scores.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

#define _FI_DISPATCH_HEAD_DIM(DIM, ...) \
  case DIM: {                           \
    constexpr uint32_t HEAD_DIM = DIM;  \
    __VA_ARGS__                         \
    break;                              \
  }

void sparse_paged_scores(TensorView q, TensorView k_cache, TensorView page_table,
                         TensorView token_to_req, TensorView query_positions,
                         TensorView sequence_lengths, TensorView visible_blocks, TensorView logits,
                         int64_t compress_ratio, double divisor) {
  CHECK_DEVICE(q, logits);
  CHECK_DEVICE(k_cache, logits);
  CHECK_DEVICE(page_table, logits);
  CHECK_DEVICE(token_to_req, logits);
  CHECK_DEVICE(query_positions, logits);
  CHECK_DEVICE(sequence_lengths, logits);
  CHECK_DEVICE(visible_blocks, logits);
  CHECK_DIM(3, q);        // [rows, heads, head_dim]
  CHECK_DIM(3, k_cache);  // [pages, page_size, head_dim]
  CHECK_DIM(2, page_table);
  CHECK_DIM(2, logits);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_cache);
  CHECK_CONTIGUOUS(token_to_req);
  CHECK_CONTIGUOUS(query_positions);
  CHECK_CONTIGUOUS(sequence_lengths);
  CHECK_CONTIGUOUS(visible_blocks);
  // The kernel walks a row of each of these with a stride of one element.
  CHECK_LAST_DIM_CONTIGUOUS(page_table);
  CHECK_LAST_DIM_CONTIGUOUS(logits);

  const int64_t rows = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t num_columns = logits.size(1);
  // Narrowed to uint32 for the kernel, where a value past that wraps: 2^32
  // would arrive as a compression ratio of zero and divide by it.
  TVM_FFI_ICHECK_GT(compress_ratio, 0) << "compress_ratio must be positive";
  TVM_FFI_ICHECK_LE(compress_ratio, 4294967295LL) << "compress_ratio must fit in 32 bits";
  TVM_FFI_ICHECK_GT(divisor, 0.0) << "divisor must be positive";
  TVM_FFI_ICHECK_EQ(k_cache.size(2), head_dim) << "cache and query head_dim must match";
  // A page holds at least one entry: the kernel divides a column by this to
  // reach its page, and a cache with no pages at all is still a page size.
  TVM_FFI_ICHECK_GT(k_cache.size(1), 0) << "a page holds at least one entry";
  TVM_FFI_ICHECK_EQ(logits.size(0), rows);
  TVM_FFI_ICHECK_EQ(token_to_req.size(0), rows);
  TVM_FFI_ICHECK_EQ(query_positions.size(0), rows);
  TVM_FFI_ICHECK_EQ(visible_blocks.size(0), rows);
  TVM_FFI_ICHECK_EQ(sequence_lengths.size(0), page_table.size(0));
  TVM_FFI_ICHECK_EQ(logits.dtype(), dl_float32) << "logits must be float32";
  TVM_FFI_ICHECK_EQ(k_cache.dtype(), q.dtype()) << "cache and query dtype must match";
  // Every index tensor is read through one pointer type.
  TVM_FFI_ICHECK_EQ(token_to_req.dtype(), page_table.dtype());
  TVM_FFI_ICHECK_EQ(query_positions.dtype(), page_table.dtype());
  TVM_FFI_ICHECK_EQ(sequence_lengths.dtype(), page_table.dtype());
  TVM_FFI_ICHECK_EQ(visible_blocks.dtype(), page_table.dtype());

  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  const cudaStream_t stream = get_stream(logits.device());
  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(page_table.dtype(), c_idtype, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), c_type, [&] {
      cudaError_t status = cudaErrorInvalidValue;
      auto launch = [&](auto head_dim_tag) {
        constexpr uint32_t HEAD_DIM = decltype(head_dim_tag)::value;
        status = SparsePagedScores<HEAD_DIM, c_type, c_idtype>(
            static_cast<const c_type*>(q.data_ptr()),
            static_cast<const c_type*>(k_cache.data_ptr()),
            static_cast<const c_idtype*>(page_table.data_ptr()),
            static_cast<const c_idtype*>(token_to_req.data_ptr()),
            static_cast<const c_idtype*>(query_positions.data_ptr()),
            static_cast<const c_idtype*>(sequence_lengths.data_ptr()),
            static_cast<c_idtype*>(visible_blocks.data_ptr()),
            static_cast<float*>(logits.data_ptr()), static_cast<uint32_t>(q.stride(0)),
            static_cast<uint32_t>(q.stride(1)), static_cast<uint32_t>(k_cache.stride(0)),
            static_cast<uint32_t>(k_cache.stride(1)), static_cast<uint32_t>(page_table.stride(0)),
            static_cast<uint32_t>(logits.stride(0)), static_cast<uint32_t>(rows),
            static_cast<uint32_t>(num_columns), static_cast<uint32_t>(k_cache.size(0)),
            static_cast<uint32_t>(page_table.size(0)), static_cast<uint32_t>(page_table.size(1)),
            static_cast<uint32_t>(num_heads), static_cast<uint32_t>(k_cache.size(1)),
            static_cast<uint32_t>(compress_ratio), static_cast<float>(divisor), stream);
      };
      switch (head_dim) {
        case 64:
          launch(std::integral_constant<uint32_t, 64>{});
          break;
        case 128:
          launch(std::integral_constant<uint32_t, 128>{});
          break;
        case 192:
          launch(std::integral_constant<uint32_t, 192>{});
          break;
        case 256:
          launch(std::integral_constant<uint32_t, 256>{});
          break;
        default:
          TVM_FFI_ICHECK(false) << "unsupported head_dim " << head_dim
                                << "; expected 64, 128, 192 or 256";
      }
      TVM_FFI_ICHECK(status != cudaErrorInvalidValue)
          << "unsupported query head count " << num_heads << "; expected at most 16";
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "SparsePagedScores failed: " << cudaGetErrorString(status);
      return true;
    });
  });
}
