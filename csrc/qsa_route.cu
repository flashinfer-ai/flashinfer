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
#include <flashinfer/attention/qsa/route.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Optional;

void qsa_expand_block_route(TensorView indexer_block_ids, TensorView query_positions,
                            TensorView seq_lens, TensorView token_to_request, TensorView out,
                            int64_t compress_ratio) {
  CHECK_DEVICE(indexer_block_ids, out);
  CHECK_DEVICE(query_positions, out);
  CHECK_DEVICE(seq_lens, out);
  CHECK_DEVICE(token_to_request, out);
  CHECK_DIM(2, indexer_block_ids);
  CHECK_DIM(2, out);
  CHECK_DIM(1, query_positions);
  CHECK_DIM(1, seq_lens);
  CHECK_DIM(1, token_to_request);
  CHECK_CONTIGUOUS(query_positions);
  CHECK_CONTIGUOUS(seq_lens);
  CHECK_CONTIGUOUS(token_to_request);

  const int64_t rows = indexer_block_ids.size(0);
  const int64_t block_topk = indexer_block_ids.size(1);
  const int64_t output_width = out.size(1);
  // These are narrowed to uint32 for the kernel, so a value past that wraps:
  // 2^32 would arrive as a page size or a compression ratio of zero.
  constexpr int64_t kUint32Max = 4294967295LL;
  TVM_FFI_ICHECK_GT(compress_ratio, 0) << "compress_ratio must be positive";
  TVM_FFI_ICHECK_LE(compress_ratio, kUint32Max) << "compress_ratio must fit in 32 bits";
  TVM_FFI_ICHECK_EQ(out.size(0), rows) << "route must have one row per query";
  TVM_FFI_ICHECK_EQ(query_positions.size(0), rows) << "one query position per row";
  TVM_FFI_ICHECK_EQ(token_to_request.size(0), rows) << "one request index per row";
  TVM_FFI_ICHECK_GT(seq_lens.size(0), 0) << "seq_lens must be non-empty";
  // The tail of the query's own block never exceeds compress_ratio - 1 tokens.
  TVM_FFI_ICHECK_EQ(output_width, block_topk * compress_ratio + compress_ratio - 1)
      << "route width must be block_topk * compress_ratio + compress_ratio - 1, got "
      << output_width;
  TVM_FFI_ICHECK_EQ(seq_lens.dtype(), indexer_block_ids.dtype());
  TVM_FFI_ICHECK_EQ(token_to_request.dtype(), indexer_block_ids.dtype());
  TVM_FFI_ICHECK_EQ(out.dtype(), indexer_block_ids.dtype());

  ffi::CUDADeviceGuard device_guard(out.device().device_id);
  const cudaStream_t stream = get_stream(out.device());
  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(indexer_block_ids.dtype(), c_idtype, [&] {
    return DISPATCH_DLPACK_IDTYPE_TO_CTYPE(query_positions.dtype(), c_postype, [&] {
      cudaError_t status = ExpandBlockRoute<c_idtype, c_postype>(
          static_cast<const c_idtype*>(indexer_block_ids.data_ptr()),
          static_cast<const c_postype*>(query_positions.data_ptr()),
          static_cast<const c_idtype*>(seq_lens.data_ptr()),
          static_cast<const c_idtype*>(token_to_request.data_ptr()),
          static_cast<c_idtype*>(out.data_ptr()),
          static_cast<uint32_t>(indexer_block_ids.stride(0)),
          static_cast<uint32_t>(indexer_block_ids.stride(1)), static_cast<uint32_t>(out.stride(0)),
          static_cast<uint32_t>(out.stride(1)), static_cast<uint32_t>(rows),
          static_cast<uint32_t>(seq_lens.size(0)), static_cast<uint32_t>(block_topk),
          static_cast<uint32_t>(compress_ratio), static_cast<uint32_t>(output_width), stream);
      TVM_FFI_ICHECK(status != cudaErrorInvalidValue)
          << "unsupported compress_ratio " << compress_ratio << "; expected a power of two <= 32";
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "ExpandBlockRoute failed: " << cudaGetErrorString(status);
      return true;
    });
  });
}

void qsa_route_from_logical(TensorView logical, TensorView token_to_request, TensorView block_table,
                            TensorView out_route, TensorView out_mask, int64_t valid_rows,
                            int64_t page_size, int64_t num_slots, Optional<TensorView> out_indptr) {
  CHECK_DEVICE(logical, out_route);
  CHECK_DEVICE(block_table, out_route);
  CHECK_DEVICE(token_to_request, out_route);
  CHECK_DEVICE(out_mask, out_route);
  CHECK_DIM(2, logical);
  CHECK_DIM(2, block_table);
  CHECK_DIM(2, out_route);
  CHECK_DIM(1, token_to_request);
  CHECK_CONTIGUOUS(out_route);
  CHECK_CONTIGUOUS(out_mask);
  CHECK_CONTIGUOUS(token_to_request);
  // The kernel walks the row of each of these with a stride of one element.
  CHECK_LAST_DIM_CONTIGUOUS(logical);
  CHECK_LAST_DIM_CONTIGUOUS(block_table);

  const int64_t rows = out_route.size(0);
  const int64_t width = out_route.size(1);
  const int64_t mask_bytes = (width + 7) / 8;
  // These are narrowed to uint32 for the kernel, so a value past that wraps:
  // 2^32 would arrive as a page size or a compression ratio of zero.
  constexpr int64_t kUint32Max = 4294967295LL;
  TVM_FFI_ICHECK_GT(page_size, 0) << "page_size must be positive";
  TVM_FFI_ICHECK_LE(page_size, kUint32Max) << "page_size must fit in 32 bits";
  TVM_FFI_ICHECK_GT(num_slots, 0) << "num_slots must be positive";
  TVM_FFI_ICHECK_LE(num_slots, kUint32Max) << "num_slots must fit in 32 bits";
  // A slot is stored in the route's dtype, so the largest one has to fit it.
  // num_slots is a count, so that is num_slots - 1: an int32 route holds a
  // slot space of 2^31 exactly, and one more than that comes back out as a
  // negative index with its mask bit set.
  TVM_FFI_ICHECK_LE(num_slots, out_route.dtype() == dl_int32 ? 2147483648LL : kUint32Max)
      << "num_slots must fit the route dtype";
  TVM_FFI_ICHECK_GE(logical.size(0), valid_rows) << "logical route must cover every live row";
  TVM_FFI_ICHECK_EQ(logical.size(1), width) << "logical route width must match the route";
  TVM_FFI_ICHECK_GE(valid_rows, 0);
  TVM_FFI_ICHECK_LE(valid_rows, rows) << "valid_rows must not exceed the route rows";
  TVM_FFI_ICHECK_GE(token_to_request.size(0), valid_rows) << "one request index per live row";
  TVM_FFI_ICHECK_EQ(out_mask.numel(), rows * mask_bytes)
      << "mask must hold ceil(width/8) bytes per row";
  TVM_FFI_ICHECK_EQ(out_mask.dtype(), dl_uint8) << "mask must be uint8";
  TVM_FFI_ICHECK_EQ(out_route.dtype(), logical.dtype());
  TVM_FFI_ICHECK_EQ(block_table.dtype(), logical.dtype());
  TVM_FFI_ICHECK_EQ(token_to_request.dtype(), logical.dtype());
  int32_t* indptr = nullptr;
  if (out_indptr.has_value()) {
    const TensorView& t = out_indptr.value();
    CHECK_DEVICE(t, out_route);
    CHECK_CONTIGUOUS(t);
    CHECK_DIM(1, t);
    TVM_FFI_ICHECK(t.dtype() == dl_int32) << "out_indptr must be int32";
    TVM_FFI_ICHECK_EQ(t.size(0), rows + 1) << "out_indptr must hold rows + 1 entries";
    // Its last entry is rows * width, written as an int32.
    TVM_FFI_ICHECK_LE(rows * width, 2147483647LL) << "rows * width must fit in an int32 indptr";
    indptr = static_cast<int32_t*>(t.data_ptr());
  }

  ffi::CUDADeviceGuard device_guard(out_route.device().device_id);
  const cudaStream_t stream = get_stream(out_route.device());
  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(logical.dtype(), c_idtype, [&] {
    cudaError_t status = QSARouteFromLogical<c_idtype>(
        static_cast<const c_idtype*>(logical.data_ptr()),
        static_cast<const c_idtype*>(token_to_request.data_ptr()),
        static_cast<const c_idtype*>(block_table.data_ptr()),
        static_cast<c_idtype*>(out_route.data_ptr()), static_cast<uint8_t*>(out_mask.data_ptr()),
        indptr, static_cast<uint32_t>(logical.stride(0)),
        static_cast<uint32_t>(block_table.stride(0)), static_cast<uint32_t>(rows),
        static_cast<uint32_t>(valid_rows), static_cast<uint32_t>(block_table.size(0)),
        static_cast<uint32_t>(width), static_cast<uint32_t>(block_table.size(1)),
        static_cast<uint32_t>(page_size), static_cast<uint32_t>(num_slots),
        static_cast<uint32_t>(mask_bytes), stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "QSARouteFromLogical failed: " << cudaGetErrorString(status);
    return true;
  });
}
