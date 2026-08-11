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

// FlashInfer JIT-compiles one fatbin for every arch in FLASHINFER_CUDA_ARCH_LIST, but
// DeviceBatchedTopK's tie-break configurations require SM 9.0+ and would static_assert on
// older targets. Defer that diagnosis to runtime (cudaErrorNotSupported); the Python
// dispatcher gates on compute capability so supported devices never see it.
#define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/std/__execution/env.h>

#include <cub/device/device_batched_topk.cuh>
#include <cuda/argument>
#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/std/limits>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace {

// cub::DeviceBatchedTopK requires a compile-time upper bound on the segment size, so the
// runtime max_len is dispatched over a small ladder of bounds (see
// CUBBatchedTopKDispatchBound). Every tier multiplies kernel instantiations (tiers x dtypes x
// requirement configs), so the ladder is kept minimal: one tier per capability class.
//
// 2^21 is DeviceBatchedTopK's own per-segment limit — it static_asserts on anything larger
// ("larger segments are future work" per the CUB docs). So this is both our top tier and the
// most this backend can ever handle; the Python dispatcher has to send longer rows to the
// radix backend.
constexpr int64_t CUB_TOPK_MAX_LEN = int64_t{1} << 21;

// Fused page-table transform epilogue, composed from stock cuda:: iterators. The batched
// API's outer output iterator is dereferenced on the device once per segment; the inner
// iterator applies the compact-page translation at the moment CUB stores each winning
// window-local index — no epilogue kernel:
//   translated = src_page_table[row, idx >> page_bits] << page_bits | (idx & (page_size - 1))
// (page_size == 1 << page_bits, mirroring the native kernel's shift form; page_size == 1
// degenerates to a plain per-position lookup).
//
// Two inner shapes, one per output mode:
//   - translated only:  transform_output_iterator(out_row, translate)
//   - translated + raw: transform_output_iterator(zip(out_row, raw_row), translate_dual)
//     — CUB stores one scalar; the functor fans it into a (translated, raw) tuple and the
//     zip reference scatters the components to the two buffers.
// The modes are distinct iterator types, so each is its own kernel instantiation (a
// deliberate trade: pure stock components over the single-instantiation hand-rolled proxy).
// The keys (score values) are not returned by the transform API and are swallowed by
// per-row discard_iterators. Write-only outputs are safe: both agents only ever assign
// through the output iterators (BlockStore is store-only; the cluster agent's three write
// sites never read back).
// Write functor: translates one winning window-local index; with raw indices requested it
// returns a (translated, raw) tuple for the zip iterator to scatter.
template <bool WithRawIndices>
struct CUBPageTranslate {
  const int32_t* page_row;
  uint32_t page_bits;
  __device__ auto operator()(int32_t idx) const {
    const int32_t page = page_row[idx >> page_bits];  // page_table[idx / page_size]
    const int32_t page_base = page << page_bits;      // page * page_size
    const int32_t page_offset = idx & ((int32_t{1} << page_bits) - 1);  // idx % page_size
    const int32_t translated = page_base | page_offset;
    if constexpr (WithRawIndices) {
      return cuda::std::tuple<int32_t, int32_t>{translated, idx};
    } else {
      return translated;
    }
  }
};

// Outer functor: dereferenced on the device once per segment, builds row `i`'s writer.
// The output mode is a compile-time switch so each mode keeps its own iterator type (and
// kernel instantiation) while sharing one definition; raw_base is unused (pass nullptr)
// when WithRawIndices is false.
template <bool WithRawIndices>
struct CUBMakePageTableRowOut {
  int32_t* out_base;
  int32_t* raw_base;
  const int32_t* src_page_table;
  int64_t page_stride;
  uint32_t page_bits;
  int64_t top_k;
  __host__ __device__ auto operator()(int64_t row) const {
    const CUBPageTranslate<WithRawIndices> translate{src_page_table + row * page_stride, page_bits};
    if constexpr (WithRawIndices) {
      return cuda::make_transform_output_iterator(
          cuda::make_zip_iterator(out_base + row * top_k, raw_base + row * top_k), translate);
    } else {
      return cuda::make_transform_output_iterator(out_base + row * top_k, translate);
    }
  }
};

// When query_bytes_out is non-null, only the workspace-size query runs (nothing is
// launched); the result is written there and the outputs/workspace are not touched.
template <int64_t MAX_LEN_BOUND, typename DType, typename ValuesOutItItT, typename RequirementsT>
cudaError_t CUBBatchedTopKRun(const DType* input, int64_t row_stride, ValuesOutItItT d_values_out,
                              const int32_t* lengths, Optional<TensorView>& maybe_workspace_buffer,
                              int64_t num_rows, int64_t max_len, int64_t top_k,
                              const RequirementsT& requirements, size_t* query_bytes_out,
                              cudaStream_t stream) {
  // Per-segment iterator over the dense (num_rows, max_len) input: d_keys_in[i] yields a
  // pointer to row i. The values-out iterator comes from the caller (the fused page-table
  // writers).
  auto d_keys_in = cuda::make_strided_iterator(cuda::make_counting_iterator(input), row_stride);
  // The keys (score values) are not returned by the transform API: every segment gets the
  // same discard iterator (the outer constant level satisfies the iterator-of-iterators
  // contract; a bare discard_iterator's reference is not an iterator and does not compile).
  auto d_keys_out = cuda::make_constant_iterator(cuda::discard_iterator{});
  // The "values" carried alongside each key are the per-segment item indices [0, max_len),
  // synthesized by a counting iterator; d_values_out then receives the top-k source indices.
  auto d_values_in = cuda::make_constant_iterator(cuda::make_counting_iterator(int32_t{0}));

  auto k_arg = cuda::args::immediate{top_k, cuda::args::bounds<int64_t{1}, MAX_LEN_BOUND>()};
  auto num_segs = cuda::args::immediate{num_rows};

  auto env = cuda::std::execution::env{requirements, cuda::stream_ref{stream}};

  // Two-phase workspace flow (CUB's "temporary storage"): size query, then run.
  auto run_with = [&](auto segment_sizes) -> cudaError_t {
    size_t workspace_bytes = 0;
    if (const auto error = cub::DeviceBatchedTopK::MaxPairs(nullptr, workspace_bytes, d_keys_in,
                                                            d_keys_out, d_values_in, d_values_out,
                                                            segment_sizes, k_arg, num_segs, env)) {
      return error;
    }
    if (query_bytes_out != nullptr) {
      *query_bytes_out = workspace_bytes;
      return cudaSuccess;
    }

    void* d_workspace = nullptr;
    bool owned = false;
    if (maybe_workspace_buffer.has_value()) {
      const auto& workspace = maybe_workspace_buffer.value();
      const size_t provided_bytes =
          static_cast<size_t>(workspace.numel()) * get_element_size(workspace);
      TVM_FFI_ICHECK(provided_bytes >= workspace_bytes)
          << "cub_topk workspace too small: need " << workspace_bytes << " bytes, have "
          << provided_bytes;
      d_workspace = workspace.data_ptr();
    } else {
      if (const auto error = cudaMallocAsync(&d_workspace, workspace_bytes, stream)) {
        return error;
      }
      owned = true;
    }

    // No early return below this point: the owned workspace must be freed on every path.
    cudaError_t status = cub::DeviceBatchedTopK::MaxPairs(d_workspace, workspace_bytes, d_keys_in,
                                                          d_keys_out, d_values_in, d_values_out,
                                                          segment_sizes, k_arg, num_segs, env);

    if (owned) {
      if (const auto free_error = cudaFreeAsync(d_workspace, stream)) {
        // Prefer the MaxPairs error over the free error when both fail.
        return status == cudaSuccess ? free_error : status;
      }
    }
    return status;
  };

  // Per-row segment sizes, read on device in stream order. The second, *runtime* bound is a
  // perf lever, not decoration: the host cannot read the device-side lengths when sizing the
  // launch, so without it CUB would size the cluster launch from the static MAX_LEN_BOUND
  // ceiling, failing single-CTA eligibility and forcing the wide multi-CTA path for every
  // segment. Passing max_len as the runtime ceiling keeps small-row launches on the cheap
  // single-CTA shape.
  // The lower bound spans the full int32 range so no lengths value can violate the bounds
  // contract (out-of-bounds values are UB): under a negative statically-known lower bound,
  // CUB clamps any negative runtime size to an empty segment (size 0), and a zero-length row
  // is a valid empty segment — CUB selects nothing for it, so with the caller's -1 prefill
  // the whole output row reads as padding. The lower bound plays no role in launch sizing
  // (only the upper bound does), so this costs nothing.
  constexpr int32_t kLengthsFloor = cuda::std::numeric_limits<int32_t>::min();
  return run_with(cuda::args::deferred_sequence{
      lengths, cuda::args::bounds<kLengthsFloor, int32_t{MAX_LEN_BOUND}>(),
      cuda::args::bounds(kLengthsFloor, static_cast<int32_t>(max_len))});
}

template <typename DType, typename ValuesOutItItT, typename RequirementsT>
cudaError_t CUBBatchedTopKDispatchBound(const DType* input, int64_t row_stride,
                                        ValuesOutItItT d_values_out, const int32_t* lengths,
                                        Optional<TensorView>& maybe_workspace_buffer,
                                        int64_t num_rows, int64_t max_len, int64_t top_k,
                                        const RequirementsT& requirements, size_t* query_bytes_out,
                                        cudaStream_t stream) {
  // CUB picks its backend from the compile-time bound: up to 8192 it can use the single-block
  // backend, which runs on any architecture. Anything larger needs the cluster backend and
  // therefore SM90+, so without this tier pre-SM90 GPUs couldn't run cub_topk at all. 8192 is
  // the largest bound the single-block backend accepts for our key/value types.
  if (max_len <= 8192) {
    return CUBBatchedTopKRun<int64_t{8192}>(input, row_stride, d_values_out, lengths,
                                            maybe_workspace_buffer, num_rows, max_len, top_k,
                                            requirements, query_bytes_out, stream);
  } else {
    // Cluster backend, SM90+ only. On an older device this doesn't crash or fall back — the
    // CUB dispatch notices at runtime and returns cudaErrorNotSupported (see the
    // CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT define at the top of this file), which
    // cub_topk() turns into an exception. The Python dispatcher is expected to route these
    // calls to the radix backend instead, so reaching this branch pre-SM90 means someone
    // forced the CUB backend explicitly.
    return CUBBatchedTopKRun<CUB_TOPK_MAX_LEN>(input, row_stride, d_values_out, lengths,
                                               maybe_workspace_buffer, num_rows, max_len, top_k,
                                               requirements, query_bytes_out, stream);
  }
}

template <typename DType, typename ValuesOutItItT>
cudaError_t CUBBatchedTopKDispatch(const DType* input, int64_t row_stride,
                                   ValuesOutItItT d_values_out, const int32_t* lengths,
                                   Optional<TensorView>& maybe_workspace_buffer, int64_t num_rows,
                                   int64_t max_len, int64_t top_k, int64_t tie_break,
                                   size_t* query_bytes_out, cudaStream_t stream) {
  namespace exec = cuda::execution;
  // Each require(...) call has a distinct type (requirements are encoded at compile time), so
  // the runtime flag must fan out into separate branches; the generic lambda factors out the
  // otherwise-identical call.
  auto run = [&](auto requirements) {
    return CUBBatchedTopKDispatchBound(input, row_stride, d_values_out, lengths,
                                       maybe_workspace_buffer, num_rows, max_len, top_k,
                                       requirements, query_bytes_out, stream);
  };

  if (tie_break == 1) {
    return run(exec::require(exec::determinism::gpu_to_gpu, exec::tie_break::prefer_smaller_index,
                             exec::output_ordering::unsorted));
  } else if (tie_break == 2) {
    return run(exec::require(exec::determinism::gpu_to_gpu, exec::tie_break::prefer_larger_index,
                             exec::output_ordering::unsorted));
  } else {
    return run(exec::require(exec::determinism::not_guaranteed, exec::tie_break::unspecified,
                             exec::output_ordering::unsorted));
  }
}

// Validation shared by the transform entry and its workspace-size query.
void CheckCUBTopKArgs(const TensorView& input, const TensorView& lengths, int64_t top_k,
                      int64_t tie_break) {
  // Rows only need to be individually contiguous: the row pitch is threaded through as
  // input.stride(0), so strided views (e.g. scores[:, :cur_len] of a wider buffer) work
  // without a .contiguous() copy.
  CHECK_CUDA(input);
  CHECK_LAST_DIM_CONTIGUOUS(input);
  CHECK_DIM(2, input);  // input: (batch_size, d)
  TVM_FFI_ICHECK(tie_break >= 0 && tie_break <= 2)
      << "Invalid tie_break mode " << tie_break
      << ", expected 0 (none), 1 (prefer small indices), or 2 (prefer large indices)";

  const int64_t num_rows = input.size(0);
  const int64_t max_len = input.size(1);
  TVM_FFI_ICHECK(top_k > 0 && top_k <= max_len)
      << "cub_topk requires 0 < top_k <= d, got top_k=" << top_k << ", d=" << max_len;
  TVM_FFI_ICHECK(max_len <= CUB_TOPK_MAX_LEN)
      << "cub_topk supports d <= " << CUB_TOPK_MAX_LEN << ", got d=" << max_len;

  CHECK_INPUT(lengths);
  CHECK_DIM(1, lengths);  // lengths: (batch_size,)
  CHECK_INPUT_TYPE(lengths, dl_int32);
  TVM_FFI_ICHECK(lengths.size(0) == num_rows)
      << "cub_topk lengths must have one entry per row: expected " << num_rows << ", got "
      << lengths.size(0);
}

}  // namespace

// CUB-backed fused top-k + page-table transform, scoped to the identity row mapping (no
// row_to_batch / row_starts / page_table_row_starts — the Python dispatcher routes those to
// the native backend). For each row i, the top-k is selected over input[i, 0:lengths[i]] and
// each winning window-local index idx is written as
//   src_page_table[i, idx / page_size] * page_size + idx % page_size
// with idx itself optionally duplicated into output_raw_indices. The translation happens
// inside CUB's own kernel via the output iterators — no epilogue launch. Rows with
// lengths[i] < top_k leave both output tails untouched; the Python wrapper pre-fills them
// with -1 (matching the native kernels, which write -1 in-kernel).
void cub_topk_page_table_transform(TensorView input, TensorView output_page_table,
                                   TensorView src_page_table, TensorView lengths,
                                   Optional<TensorView> maybe_output_raw_indices,
                                   Optional<TensorView> maybe_workspace_buffer, int64_t top_k,
                                   int64_t tie_break, int64_t page_size) {
  CheckCUBTopKArgs(input, lengths, top_k, tie_break);
  const int32_t* lengths_ptr = static_cast<const int32_t*>(lengths.data_ptr());
  CHECK_INPUT(output_page_table);
  CHECK_DIM(2, output_page_table);  // (num_rows, top_k)
  CHECK_INPUT_TYPE(output_page_table, dl_int32);
  TVM_FFI_ICHECK(output_page_table.size(0) == input.size(0) && output_page_table.size(1) == top_k)
      << "cub_topk output_page_table must have shape (num_rows, top_k) = (" << input.size(0) << ", "
      << top_k << "), got (" << output_page_table.size(0) << ", " << output_page_table.size(1)
      << ")";
  CHECK_CUDA(src_page_table);
  CHECK_LAST_DIM_CONTIGUOUS(src_page_table);
  CHECK_DIM(2, src_page_table);
  CHECK_INPUT_TYPE(src_page_table, dl_int32);
  TVM_FFI_ICHECK(src_page_table.size(0) == input.size(0))
      << "cub_topk src_page_table must have one row per input row (identity mapping): expected "
      << input.size(0) << ", got " << src_page_table.size(0);

  int32_t* output_raw_indices = nullptr;
  if (maybe_output_raw_indices.has_value()) {
    const auto& raw = maybe_output_raw_indices.value();
    CHECK_INPUT(raw);
    CHECK_DIM(2, raw);
    CHECK_INPUT_TYPE(raw, dl_int32);
    CHECK_SHAPE(raw, output_page_table);
    output_raw_indices = static_cast<int32_t*>(raw.data_ptr());
  }

  TVM_FFI_ICHECK(page_size > 0 && cuda::is_power_of_two(page_size) &&
                 page_size <= (int64_t{1} << 30))
      << "cub_topk page_size must be a power of two in [1, 2^30], got " << page_size;

  const auto page_bits = static_cast<uint32_t>(cuda::ilog2(page_size));

  const int64_t num_rows = input.size(0);
  const int64_t max_len = input.size(1);
  if (num_rows == 0) {
    return;
  }

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  auto* out_ptr = static_cast<int32_t*>(output_page_table.data_ptr());
  const auto* src_ptr = static_cast<const int32_t*>(src_page_table.data_ptr());
  const int64_t page_stride = src_page_table.stride(0);

  // The two output modes are distinct iterator types; the generic lambda serves both.
  cudaError_t status = cudaErrorInvalidValue;
  auto run = [&](auto row_out_maker) {
    auto d_values_out =
        cuda::make_transform_iterator(cuda::make_counting_iterator(int64_t{0}), row_out_maker);
    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(input.dtype(), c_type, [&] {
      status = CUBBatchedTopKDispatch(static_cast<const c_type*>(input.data_ptr()), input.stride(0),
                                      d_values_out, lengths_ptr, maybe_workspace_buffer, num_rows,
                                      max_len, top_k, tie_break,
                                      /*query_bytes_out=*/nullptr, stream);
      return true;
    });
  };
  if (output_raw_indices != nullptr) {
    run(CUBMakePageTableRowOut<true>{out_ptr, output_raw_indices, src_ptr, page_stride, page_bits,
                                     top_k});
  } else {
    run(CUBMakePageTableRowOut<false>{out_ptr, nullptr, src_ptr, page_stride, page_bits, top_k});
  }

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk_page_table_transform failed with error code " << cudaGetErrorString(status);
}

// Workspace query for the transform variant. Must instantiate the same dispatch (same
// iterator types) as the run so the byte count matches the kernel that will execute — the
// output mode changes the iterator type, so the caller states it via with_raw_indices.
int64_t cub_topk_page_table_transform_workspace_size(TensorView input, TensorView lengths,
                                                     int64_t top_k, int64_t tie_break,
                                                     bool with_raw_indices) {
  CheckCUBTopKArgs(input, lengths, top_k, tie_break);
  const int32_t* lengths_ptr = static_cast<const int32_t*>(lengths.data_ptr());

  const int64_t num_rows = input.size(0);
  const int64_t max_len = input.size(1);
  if (num_rows == 0) {
    return 0;
  }

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  // Types-only query: iterator bases are never dereferenced.
  Optional<TensorView> no_workspace;
  size_t workspace_bytes = 0;
  cudaError_t status = cudaErrorInvalidValue;
  auto query = [&](auto row_out_maker) {
    auto d_values_out =
        cuda::make_transform_iterator(cuda::make_counting_iterator(int64_t{0}), row_out_maker);
    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(input.dtype(), c_type, [&] {
      status = CUBBatchedTopKDispatch(static_cast<const c_type*>(input.data_ptr()), input.stride(0),
                                      d_values_out, lengths_ptr, no_workspace, num_rows, max_len,
                                      top_k, tie_break, &workspace_bytes, stream);
      return true;
    });
  };
  if (with_raw_indices) {
    query(CUBMakePageTableRowOut<true>{nullptr, nullptr, nullptr, 0, 0, top_k});
  } else {
    query(CUBMakePageTableRowOut<false>{nullptr, nullptr, nullptr, 0, 0, top_k});
  }

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk_page_table_transform workspace-size query failed with error code "
      << cudaGetErrorString(status);
  return static_cast<int64_t>(workspace_bytes);
}
