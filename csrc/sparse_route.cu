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
#include <flashinfer/sparse_route.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

void expand_block_route(TensorView block_indices, TensorView query_positions,
                        TensorView sequence_lengths, TensorView token_to_req, TensorView out,
                        int64_t compress_ratio) {
  CHECK_DEVICE(block_indices, out);
  CHECK_DEVICE(query_positions, out);
  CHECK_DEVICE(sequence_lengths, out);
  CHECK_DEVICE(token_to_req, out);
  CHECK_DIM(2, block_indices);
  CHECK_DIM(2, out);
  CHECK_DIM(1, query_positions);
  CHECK_DIM(1, sequence_lengths);
  CHECK_DIM(1, token_to_req);
  CHECK_CONTIGUOUS(query_positions);
  CHECK_CONTIGUOUS(sequence_lengths);
  CHECK_CONTIGUOUS(token_to_req);

  const int64_t rows = block_indices.size(0);
  const int64_t block_topk = block_indices.size(1);
  const int64_t output_width = out.size(1);
  TVM_FFI_ICHECK_GT(compress_ratio, 0) << "compress_ratio must be positive";
  TVM_FFI_ICHECK_EQ(out.size(0), rows) << "route must have one row per query";
  TVM_FFI_ICHECK_EQ(query_positions.size(0), rows) << "one query position per row";
  TVM_FFI_ICHECK_EQ(token_to_req.size(0), rows) << "one request index per row";
  TVM_FFI_ICHECK_GT(sequence_lengths.size(0), 0) << "sequence_lengths must be non-empty";
  // The tail of the query's own block never exceeds compress_ratio - 1 tokens.
  TVM_FFI_ICHECK_EQ(output_width, block_topk * compress_ratio + compress_ratio - 1)
      << "route width must be block_topk * compress_ratio + compress_ratio - 1, got "
      << output_width;
  TVM_FFI_ICHECK_EQ(query_positions.dtype(), block_indices.dtype());
  TVM_FFI_ICHECK_EQ(sequence_lengths.dtype(), block_indices.dtype());
  TVM_FFI_ICHECK_EQ(token_to_req.dtype(), block_indices.dtype());
  TVM_FFI_ICHECK_EQ(out.dtype(), block_indices.dtype());

  ffi::CUDADeviceGuard device_guard(out.device().device_id);
  const cudaStream_t stream = get_stream(out.device());
  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(block_indices.dtype(), c_idtype, [&] {
    cudaError_t status = ExpandBlockRoute<c_idtype>(
        static_cast<const c_idtype*>(block_indices.data_ptr()),
        static_cast<const c_idtype*>(query_positions.data_ptr()),
        static_cast<const c_idtype*>(sequence_lengths.data_ptr()),
        static_cast<const c_idtype*>(token_to_req.data_ptr()),
        static_cast<c_idtype*>(out.data_ptr()), static_cast<uint32_t>(block_indices.stride(0)),
        static_cast<uint32_t>(block_indices.stride(1)), static_cast<uint32_t>(out.stride(0)),
        static_cast<uint32_t>(out.stride(1)), static_cast<uint32_t>(rows),
        static_cast<uint32_t>(sequence_lengths.size(0)), static_cast<uint32_t>(block_topk),
        static_cast<uint32_t>(compress_ratio), static_cast<uint32_t>(output_width), stream);
    TVM_FFI_ICHECK(status != cudaErrorInvalidValue)
        << "unsupported compress_ratio " << compress_ratio << "; expected a power of two <= 32";
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "ExpandBlockRoute failed: " << cudaGetErrorString(status);
    return true;
  });
}

void qsa_route_from_blocks(TensorView block_indices, TensorView query_positions,
                           TensorView sequence_lengths, TensorView token_to_req,
                           TensorView block_table, TensorView out_logical, TensorView out_route,
                           TensorView out_mask, int64_t compress_ratio, int64_t page_size,
                           int64_t num_slots) {
  CHECK_DEVICE(block_indices, out_route);
  CHECK_DEVICE(block_table, out_route);
  CHECK_DEVICE(query_positions, out_route);
  CHECK_DEVICE(sequence_lengths, out_route);
  CHECK_DEVICE(token_to_req, out_route);
  CHECK_DEVICE(out_logical, out_route);
  CHECK_DEVICE(out_mask, out_route);
  CHECK_DIM(2, block_indices);
  CHECK_DIM(2, block_table);
  CHECK_DIM(2, out_logical);
  CHECK_DIM(2, out_route);
  CHECK_DIM(1, query_positions);
  CHECK_DIM(1, sequence_lengths);
  CHECK_DIM(1, token_to_req);
  CHECK_CONTIGUOUS(out_route);
  CHECK_CONTIGUOUS(out_mask);
  CHECK_CONTIGUOUS(query_positions);
  CHECK_CONTIGUOUS(sequence_lengths);
  CHECK_CONTIGUOUS(token_to_req);
  // The kernel walks the row of each of these with a stride of one element.
  CHECK_LAST_DIM_CONTIGUOUS(out_logical);
  CHECK_LAST_DIM_CONTIGUOUS(block_table);

  const int64_t rows = block_indices.size(0);
  const int64_t block_topk = block_indices.size(1);
  const int64_t output_width = block_topk * compress_ratio + compress_ratio - 1;
  const int64_t mask_bytes = (output_width + 7) / 8;
  TVM_FFI_ICHECK_GT(compress_ratio, 0) << "compress_ratio must be positive";
  TVM_FFI_ICHECK_GT(page_size, 0) << "page_size must be positive";
  TVM_FFI_ICHECK_GT(num_slots, 0) << "num_slots must be positive";
  TVM_FFI_ICHECK_EQ(out_route.size(0), rows);
  TVM_FFI_ICHECK_EQ(out_route.size(1), output_width);
  TVM_FFI_ICHECK_EQ(out_logical.size(1), output_width);
  TVM_FFI_ICHECK_GE(out_logical.size(0), rows);
  TVM_FFI_ICHECK_EQ(out_mask.numel(), rows * mask_bytes)
      << "mask must hold ceil(width/8) bytes per row";
  TVM_FFI_ICHECK_EQ(out_mask.dtype(), dl_uint8) << "mask must be uint8";
  TVM_FFI_ICHECK_EQ(query_positions.size(0), rows);
  TVM_FFI_ICHECK_EQ(token_to_req.size(0), rows);
  // Every index tensor is read through one pointer type.
  TVM_FFI_ICHECK_EQ(out_logical.dtype(), block_indices.dtype());
  TVM_FFI_ICHECK_EQ(out_route.dtype(), block_indices.dtype());
  TVM_FFI_ICHECK_EQ(block_table.dtype(), block_indices.dtype());
  TVM_FFI_ICHECK_EQ(query_positions.dtype(), block_indices.dtype());
  TVM_FFI_ICHECK_EQ(sequence_lengths.dtype(), block_indices.dtype());
  TVM_FFI_ICHECK_EQ(token_to_req.dtype(), block_indices.dtype());
  // A row indexes both by the same request, so a shorter table would be read
  // past its end.
  TVM_FFI_ICHECK_EQ(block_table.size(0), sequence_lengths.size(0))
      << "block table and sequence lengths must cover the same requests";

  ffi::CUDADeviceGuard device_guard(out_route.device().device_id);
  const cudaStream_t stream = get_stream(out_route.device());
  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(block_indices.dtype(), c_idtype, [&] {
    cudaError_t status = QSARouteFromBlocks<c_idtype>(
        static_cast<const c_idtype*>(block_indices.data_ptr()),
        static_cast<const c_idtype*>(query_positions.data_ptr()),
        static_cast<const c_idtype*>(sequence_lengths.data_ptr()),
        static_cast<const c_idtype*>(token_to_req.data_ptr()),
        static_cast<const c_idtype*>(block_table.data_ptr()),
        static_cast<c_idtype*>(out_logical.data_ptr()),
        static_cast<c_idtype*>(out_route.data_ptr()), static_cast<uint8_t*>(out_mask.data_ptr()),
        static_cast<uint32_t>(block_indices.stride(0)),
        static_cast<uint32_t>(block_indices.stride(1)),
        static_cast<uint32_t>(out_logical.stride(0)), static_cast<uint32_t>(block_table.stride(0)),
        static_cast<uint32_t>(rows), static_cast<uint32_t>(sequence_lengths.size(0)),
        static_cast<uint32_t>(block_topk), static_cast<uint32_t>(block_table.size(1)),
        static_cast<uint32_t>(page_size), static_cast<uint32_t>(num_slots),
        static_cast<uint32_t>(mask_bytes), static_cast<uint32_t>(compress_ratio), stream);
    TVM_FFI_ICHECK(status != cudaErrorInvalidValue)
        << "unsupported compress_ratio " << compress_ratio << "; expected a power of two <= 32";
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "QSARouteFromBlocks failed: " << cudaGetErrorString(status);
    return true;
  });
}

void qsa_route_from_logical(TensorView logical, TensorView token_to_req, TensorView block_table,
                            TensorView out_route, TensorView out_mask, int64_t valid_rows,
                            int64_t page_size, int64_t num_slots) {
  CHECK_DEVICE(logical, out_route);
  CHECK_DEVICE(block_table, out_route);
  CHECK_DEVICE(token_to_req, out_route);
  CHECK_DEVICE(out_mask, out_route);
  CHECK_DIM(2, logical);
  CHECK_DIM(2, block_table);
  CHECK_DIM(2, out_route);
  CHECK_DIM(1, token_to_req);
  CHECK_CONTIGUOUS(out_route);
  CHECK_CONTIGUOUS(out_mask);
  CHECK_CONTIGUOUS(token_to_req);
  // The kernel walks the row of each of these with a stride of one element.
  CHECK_LAST_DIM_CONTIGUOUS(logical);
  CHECK_LAST_DIM_CONTIGUOUS(block_table);

  const int64_t rows = out_route.size(0);
  const int64_t width = out_route.size(1);
  const int64_t mask_bytes = (width + 7) / 8;
  TVM_FFI_ICHECK_GT(page_size, 0) << "page_size must be positive";
  TVM_FFI_ICHECK_GT(num_slots, 0) << "num_slots must be positive";
  TVM_FFI_ICHECK_GE(logical.size(0), valid_rows) << "logical route must cover every live row";
  TVM_FFI_ICHECK_EQ(logical.size(1), width) << "logical route width must match the route";
  TVM_FFI_ICHECK_GE(valid_rows, 0);
  TVM_FFI_ICHECK_LE(valid_rows, rows) << "valid_rows must not exceed the route rows";
  TVM_FFI_ICHECK_GE(token_to_req.size(0), valid_rows) << "one request index per live row";
  TVM_FFI_ICHECK_EQ(out_mask.numel(), rows * mask_bytes)
      << "mask must hold ceil(width/8) bytes per row";
  TVM_FFI_ICHECK_EQ(out_mask.dtype(), dl_uint8) << "mask must be uint8";
  TVM_FFI_ICHECK_EQ(out_route.dtype(), logical.dtype());
  TVM_FFI_ICHECK_EQ(block_table.dtype(), logical.dtype());
  TVM_FFI_ICHECK_EQ(token_to_req.dtype(), logical.dtype());

  ffi::CUDADeviceGuard device_guard(out_route.device().device_id);
  const cudaStream_t stream = get_stream(out_route.device());
  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(logical.dtype(), c_idtype, [&] {
    cudaError_t status = QSARouteFromLogical<c_idtype>(
        static_cast<const c_idtype*>(logical.data_ptr()),
        static_cast<const c_idtype*>(token_to_req.data_ptr()),
        static_cast<const c_idtype*>(block_table.data_ptr()),
        static_cast<c_idtype*>(out_route.data_ptr()), static_cast<uint8_t*>(out_mask.data_ptr()),
        static_cast<uint32_t>(logical.stride(0)), static_cast<uint32_t>(block_table.stride(0)),
        static_cast<uint32_t>(rows), static_cast<uint32_t>(valid_rows),
        static_cast<uint32_t>(block_table.size(0)), static_cast<uint32_t>(width),
        static_cast<uint32_t>(block_table.size(1)), static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(num_slots), static_cast<uint32_t>(mask_bytes), stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "QSARouteFromLogical failed: " << cudaGetErrorString(status);
    return true;
  });
}
