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
#pragma once

#include "tvm_ffi_utils.h"

// Shared argument validation for the fused top-k + page-table transform launchers
// (radix_topk_page_table_transform and cub_topk_page_table_transform), so the backends
// accept and reject inputs identically. This is the union of both launchers' historical
// checks; backend-specific constraints (tie-break encoding, CUB's segment-size cap, the
// radix row-states buffer) stay in the respective launchers.
//
// Everything host-checkable is checked here. Device-resident *values* (lengths windows,
// row_to_batch targets, start offsets) cannot be validated without a sync and are caller
// contracts: out-of-range values are undefined behavior.
//
// src_page_table only needs contiguous rows (its row stride is threaded through to both
// backends), matching the input's requirement.
inline void CheckPageTableTransformArgs(
    const tvm::ffi::TensorView& input, const tvm::ffi::TensorView& output_page_table,
    const tvm::ffi::TensorView& src_page_table, const tvm::ffi::TensorView& lengths,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& maybe_row_to_batch,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& maybe_row_starts,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& maybe_page_table_row_starts,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& maybe_output_raw_indices, int64_t top_k,
    int64_t page_size) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_INPUT_AND_TYPE(output_page_table, dl_int32);
  CHECK_CUDA(src_page_table);
  CHECK_LAST_DIM_CONTIGUOUS(src_page_table);
  CHECK_INPUT_TYPE(src_page_table, dl_int32);
  CHECK_INPUT_AND_TYPE(lengths, dl_int32);
  CHECK_DEVICE(output_page_table, input);
  CHECK_DEVICE(src_page_table, input);
  CHECK_DEVICE(lengths, input);
  CHECK_DIM(2, input);              // input: (num_rows, max_len)
  CHECK_DIM(2, output_page_table);  // output_page_table: (num_rows, top_k)
  CHECK_DIM(2, src_page_table);     // src_page_table: (batch_size, max_page_table_length)
  CHECK_DIM(1, lengths);            // lengths: (num_rows,)

  const int64_t num_rows = input.size(0);
  TVM_FFI_ICHECK_GT(top_k, 0) << "top_k must be positive, got " << top_k;
  TVM_FFI_ICHECK_EQ(output_page_table.size(0), num_rows)
      << "output_page_table must have shape (num_rows, top_k)";
  TVM_FFI_ICHECK_EQ(output_page_table.size(1), top_k)
      << "output_page_table must have shape (num_rows, top_k)";
  TVM_FFI_ICHECK_EQ(lengths.size(0), num_rows) << "lengths must have shape (num_rows,)";
  TVM_FFI_ICHECK_GE(input.stride(0), input.size(1)) << "input rows must not overlap";
  if (!maybe_row_to_batch.has_value()) {
    // Identity mapping: score row i gathers through page-table row i, so the table must
    // cover every row. With row_to_batch the table has one row per request (fewer than
    // score rows); its values are device-resident and unvalidatable here.
    TVM_FFI_ICHECK_EQ(src_page_table.size(0), num_rows)
        << "src_page_table must have one row per input row (identity mapping)";
  }

  TVM_FFI_ICHECK_GT(page_size, 0) << "page_size must be positive";
  TVM_FFI_ICHECK_EQ(page_size & (page_size - 1), 0) << "page_size must be a power of two";
  TVM_FFI_ICHECK_LE(page_size, static_cast<int64_t>(1) << 30) << "page_size must not exceed 2^30";
  // With compact pages the table-slot base cannot be derived from the token-unit
  // row_starts fallback.
  TVM_FFI_ICHECK(
      !(page_size > 1 && maybe_row_starts.has_value() && !maybe_page_table_row_starts.has_value()))
      << "page_table_row_starts is required with page_size > 1 and row_starts";

  // Optional per-row int32 metadata: one entry per score row.
  const auto check_per_row_metadata = [&](const tvm::ffi::Optional<tvm::ffi::TensorView>& maybe_t,
                                          const char* name) {
    if (!maybe_t.has_value()) {
      return;
    }
    const auto& t = maybe_t.value();
    CHECK_INPUT_AND_TYPE(t, dl_int32);
    CHECK_DEVICE(t, input);
    CHECK_DIM(1, t);
    TVM_FFI_ICHECK_EQ(t.size(0), num_rows)
        << name << " must have one entry per row: expected " << num_rows << ", got " << t.size(0);
  };
  check_per_row_metadata(maybe_row_to_batch, "row_to_batch");
  check_per_row_metadata(maybe_row_starts, "row_starts");
  check_per_row_metadata(maybe_page_table_row_starts, "page_table_row_starts");

  if (maybe_output_raw_indices.has_value()) {
    const auto& raw = maybe_output_raw_indices.value();
    CHECK_INPUT_AND_TYPE(raw, dl_int32);
    CHECK_DEVICE(raw, input);
    CHECK_DIM(2, raw);
    CHECK_SHAPE(raw, output_page_table);
  }
}
