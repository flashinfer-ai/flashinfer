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
#include <flashinfer/sampling.cuh>
#include <flashinfer/topk.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Optional;

inline sampling::TopKTieBreak ParseTopKTieBreak(int64_t tie_break) {
  switch (tie_break) {
    case 0:
      return sampling::TopKTieBreak::None;
    case 1:
      return sampling::TopKTieBreak::Small;
    case 2:
      return sampling::TopKTieBreak::Large;
    default:
      TVM_FFI_ICHECK(false)
          << "Invalid tie_break mode " << tie_break
          << ", expected 0 (none), 1 (prefer small indices), or 2 (prefer large indices)";
      return sampling::TopKTieBreak::None;
  }
}

void radix_topk(TensorView input, TensorView output_indices, TensorView output_values,
                Optional<TensorView> maybe_row_states_buffer, int64_t top_k, bool sorted_output,
                bool deterministic, int64_t tie_break, bool dsa_graph_safe) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_indices);
  CHECK_INPUT(output_values);
  CHECK_DIM(2, input);           // input: (batch_size, d)
  CHECK_DIM(2, output_indices);  // output_indices: (batch_size, top_k)
  CHECK_DIM(2, output_values);   // output_values: (batch_size, top_k)

  unsigned int batch_size = input.size(0);
  unsigned int d = input.size(1);

  cudaSetDevice(input.device().device_id);
  auto stream = get_stream(input.device());

  cudaError_t status;
  auto dtype = input.dtype();
  sampling::TopKTieBreak tie_break_mode = ParseTopKTieBreak(tie_break);
  if (tie_break_mode != sampling::TopKTieBreak::None) {
    deterministic = true;
  }
  // Get row_states_buffer if provided (for multi-CTA path)
  sampling::RadixRowState* row_states_ptr = nullptr;
  if (maybe_row_states_buffer.has_value()) {
    row_states_ptr =
        static_cast<sampling::RadixRowState*>(maybe_row_states_buffer.value().data_ptr());
  }

  // Use unified dispatch with heuristics to choose between FilteredTopK and RadixTopK
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    status = sampling::TopKDispatch<c_type, int32_t>(
        static_cast<c_type*>(input.data_ptr()), static_cast<int32_t*>(output_indices.data_ptr()),
        static_cast<c_type*>(output_values.data_ptr()), batch_size, static_cast<uint32_t>(top_k), d,
        row_states_ptr, sorted_output, deterministic, tie_break_mode, stream, dsa_graph_safe);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopK failed with error code " << cudaGetErrorString(status);
}

void radix_topk_page_table_transform(TensorView input, TensorView output_page_table,
                                     TensorView src_page_table,
                                     Optional<TensorView> maybe_row_to_batch, TensorView lengths,
                                     Optional<TensorView> maybe_row_states_buffer, int64_t top_k,
                                     bool deterministic, int64_t tie_break, bool dsa_graph_safe,
                                     Optional<TensorView> maybe_row_starts) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_page_table);
  CHECK_INPUT(src_page_table);
  CHECK_INPUT(lengths);
  CHECK_DIM(2, input);              // input: (num_rows, max_len)
  CHECK_DIM(2, output_page_table);  // output_page_table: (num_rows, top_k)
  CHECK_DIM(2, src_page_table);     // src_page_table: (batch_size, max_len)
  CHECK_DIM(1, lengths);            // lengths: (num_rows,)
  if (maybe_row_starts.has_value()) {
    CHECK_INPUT(maybe_row_starts.value());
    CHECK_DIM(1, maybe_row_starts.value());
  }

  unsigned int num_rows = input.size(0);
  unsigned int max_len = input.size(1);
  int64_t src_stride = src_page_table.stride(0);

  cudaSetDevice(input.device().device_id);
  auto stream = get_stream(input.device());

  cudaError_t status;
  auto dtype = input.dtype();
  sampling::TopKTieBreak tie_break_mode = ParseTopKTieBreak(tie_break);
  if (tie_break_mode != sampling::TopKTieBreak::None) {
    deterministic = true;
  }

  sampling::RadixRowState* row_states_ptr = nullptr;
  if (maybe_row_states_buffer.has_value()) {
    row_states_ptr =
        static_cast<sampling::RadixRowState*>(maybe_row_states_buffer.value().data_ptr());
  }

  int32_t* row_to_batch_ptr = nullptr;
  if (maybe_row_to_batch.has_value()) {
    row_to_batch_ptr = static_cast<int32_t*>(maybe_row_to_batch.value().data_ptr());
  }
  int32_t* row_starts_ptr = nullptr;
  if (maybe_row_starts.has_value()) {
    TVM_FFI_ICHECK(static_cast<unsigned int>(maybe_row_starts.value().size(0)) == num_rows)
        << "row_starts must have shape (num_rows,)";
    row_starts_ptr = static_cast<int32_t*>(maybe_row_starts.value().data_ptr());
  }

  // Use unified dispatch with heuristics to choose between FilteredTopK and RadixTopK
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    status = sampling::TopKPageTableTransformDispatch<c_type, int32_t>(
        static_cast<c_type*>(input.data_ptr()), static_cast<int32_t*>(output_page_table.data_ptr()),
        static_cast<const int32_t*>(src_page_table.data_ptr()), src_stride,
        static_cast<int32_t*>(lengths.data_ptr()), row_starts_ptr, row_to_batch_ptr, num_rows,
        static_cast<uint32_t>(top_k), max_len, row_states_ptr, deterministic, tie_break_mode,
        stream, dsa_graph_safe);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKPageTableTransform failed with error code " << cudaGetErrorString(status);
}

void radix_topk_ragged_transform(TensorView input, TensorView output_indices, TensorView offsets,
                                 TensorView lengths, Optional<TensorView> maybe_row_states_buffer,
                                 int64_t top_k, bool deterministic, int64_t tie_break,
                                 bool dsa_graph_safe, Optional<TensorView> maybe_row_starts) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_indices);
  CHECK_INPUT(offsets);
  CHECK_INPUT(lengths);
  CHECK_DIM(2, input);           // input: (num_rows, max_len)
  CHECK_DIM(2, output_indices);  // output_indices: (num_rows, top_k)
  CHECK_DIM(1, offsets);         // offsets: (num_rows,)
  CHECK_DIM(1, lengths);         // lengths: (num_rows,)
  if (maybe_row_starts.has_value()) {
    CHECK_INPUT(maybe_row_starts.value());
    CHECK_DIM(1, maybe_row_starts.value());
  }

  unsigned int num_rows = input.size(0);
  unsigned int max_len = input.size(1);

  cudaSetDevice(input.device().device_id);
  auto stream = get_stream(input.device());

  cudaError_t status;
  auto dtype = input.dtype();
  sampling::TopKTieBreak tie_break_mode = ParseTopKTieBreak(tie_break);
  if (tie_break_mode != sampling::TopKTieBreak::None) {
    deterministic = true;
  }

  sampling::RadixRowState* row_states_ptr = nullptr;
  if (maybe_row_states_buffer.has_value()) {
    row_states_ptr =
        static_cast<sampling::RadixRowState*>(maybe_row_states_buffer.value().data_ptr());
  }
  int32_t* row_starts_ptr = nullptr;
  if (maybe_row_starts.has_value()) {
    TVM_FFI_ICHECK(static_cast<unsigned int>(maybe_row_starts.value().size(0)) == num_rows)
        << "row_starts must have shape (num_rows,)";
    row_starts_ptr = static_cast<int32_t*>(maybe_row_starts.value().data_ptr());
  }

  // Use unified dispatch with heuristics to choose between FilteredTopK and RadixTopK
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    status = sampling::TopKRaggedTransformDispatch<c_type, int32_t>(
        static_cast<c_type*>(input.data_ptr()), static_cast<int32_t*>(output_indices.data_ptr()),
        static_cast<const int32_t*>(offsets.data_ptr()), static_cast<int32_t*>(lengths.data_ptr()),
        row_starts_ptr, num_rows, static_cast<uint32_t>(top_k), max_len, row_states_ptr,
        deterministic, tie_break_mode, stream, dsa_graph_safe);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKRaggedTransform failed with error code " << cudaGetErrorString(status);
}

bool can_implement_filtered_topk() { return sampling::CanImplementFilteredTopK(); }
