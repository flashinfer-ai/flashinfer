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

#include "topk_transform_checks.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace {

// cub::DeviceBatchedTopK requires a compile-time upper bound on the segment size, so the
// runtime max_len is dispatched over a small ladder of bounds (see
// CUBBatchedTopKDispatch). Every tier multiplies kernel instantiations (tiers x
// dtypes x requirement configs), so the ladder is kept minimal: one tier per capability
// class.
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
//
// The three per-row metadata pointers are nullable (identity/zero defaults), so packed
// layouts add no instantiation axis:
//   - row_to_batch: score row i gathers through page-table row row_to_batch[i] (many score
//     rows of one request share its table row); nullptr => table row i.
//   - page_table_row_starts / row_starts: the lookup's base offset within the table row, in
//     TABLE-SLOT units (the winning index is window-local; the table row describes the whole
//     request). Fallback chain mirrors the native kernel (topk.cuh):
//     page_table_start = page_table_row_starts ? : row_starts ? : 0.
//     The offset is folded into the row's base pointer here, so the write functor is
//     oblivious to it.
template <bool WithRawIndices>
struct CUBMakePageTableRowOut {
  int32_t* out_base;
  int32_t* raw_base;
  const int32_t* src_page_table;
  const int32_t* row_to_batch;           // nullptr => table row i
  const int32_t* row_starts;             // fallback for page_table_row_starts
  const int32_t* page_table_row_starts;  // nullptr => row_starts (or 0)
  int64_t page_stride;
  uint32_t page_bits;
  int64_t top_k;
  __host__ __device__ auto operator()(int64_t row) const {
    const int64_t batch = (row_to_batch != nullptr) ? row_to_batch[row] : row;
    const int64_t page_table_start = (page_table_row_starts != nullptr) ? page_table_row_starts[row]
                                     : (row_starts != nullptr)          ? row_starts[row]
                                                                        : int64_t{0};
    const CUBPageTranslate<WithRawIndices> translate{
        src_page_table + batch * page_stride + page_table_start, page_bits};
    if constexpr (WithRawIndices) {
      return cuda::make_transform_output_iterator(
          cuda::make_zip_iterator(out_base + row * top_k, raw_base + row * top_k), translate);
    } else {
      return cuda::make_transform_output_iterator(out_base + row * top_k, translate);
    }
  }
};

// Ragged transform epilogue. Write functor: shifts one winning window-local index into the
// caller's global coordinates by the row's offset as CUB stores it — no epilogue kernel,
// mirroring the page-table path. (row_starts shifts the read window only and does NOT add
// into the output, matching the native ragged kernel.)
struct CUBRaggedTranslate {
  int32_t offset;
  __device__ int32_t operator()(int32_t idx) const { return idx + offset; }
};

// Outer functor for the ragged transform: dereferenced on the device once per segment,
// builds row `i`'s writer with the row's offset folded in (offsets[row] is read on the
// device at outer dereference, like the deferred lengths). Single output mode — the ragged
// API returns indices only.
struct CUBMakeRaggedRowOut {
  int32_t* out_base;
  const int32_t* offsets;
  int64_t top_k;
  __host__ __device__ auto operator()(int64_t row) const {
    return cuda::make_transform_output_iterator(out_base + row * top_k,
                                                CUBRaggedTranslate{offsets[row]});
  }
};

// DeviceBatchedTopK writes only min(max(length, 0), top_k) items for a variable-length
// segment. Fill the untouched suffix after selection so short and empty rows preserve the
// transform APIs' -1 padding contract.
//
// Note: this kernel writes only [valid, top_k) and reads only `lengths`, while selection
// writes [0, valid) — the two are disjoint, so the after-selection ordering is convention,
// not a data dependency (it could equally run before selection or overlap it via PDL).
constexpr int kFillTailsThreads = 256;

__global__ void __launch_bounds__(kFillTailsThreads)
    CUBFillTopKTailsKernel(int32_t* __restrict__ output, int32_t* __restrict__ output_raw_indices,
                           const int32_t* __restrict__ lengths, int64_t num_rows, int64_t top_k) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= num_rows) {
    return;
  }

  int64_t valid = static_cast<int64_t>(lengths[row]);
  valid = valid < 0 ? 0 : valid;
  valid = valid > top_k ? top_k : valid;
  for (int64_t col = valid + static_cast<int64_t>(threadIdx.x); col < top_k;
       col += static_cast<int64_t>(blockDim.x)) {
    const int64_t offset = row * top_k + col;
    output[offset] = -1;
    if (output_raw_indices != nullptr) {
      output_raw_indices[offset] = -1;
    }
  }
}

cudaError_t CUBFillTopKTails(int32_t* output, int32_t* output_raw_indices, const int32_t* lengths,
                             int64_t num_rows, int64_t top_k, cudaStream_t stream) {
  if (num_rows == 0) {
    return cudaSuccess;
  }
  CUBFillTopKTailsKernel<<<static_cast<uint32_t>(num_rows), kFillTailsThreads, 0, stream>>>(
      output, output_raw_indices, lengths, num_rows, top_k);
  return cudaGetLastError();
}

// Per-segment input rows with a per-row window start: d_keys_in[row] points at
// input[row, row_starts[row]:] (row_starts read on the device at outer dereference, like the
// deferred lengths). Used only when row_starts is present; the whole-row case uses a plain
// strided iterator (a separate instantiation). Must stay __host__ __device__:
// cuda::transform_iterator deduces its reference type via host-side invoke_result.
template <typename DType>
struct CUBMakeRowIn {
  const DType* input;
  int64_t row_stride;
  const int32_t* row_starts;
  __device__ const DType* operator()(int64_t row) const {
    return input + row * row_stride + row_starts[row];
  }
};

// One cub::DeviceBatchedTopK::MaxPairs invocation with the two-phase workspace flow (CUB's
// "temporary storage"): size query first, then the run against the provided (or
// stream-allocated) workspace. Fully generic in the data iterators and size arguments —
// everything API-specific is prepared by the caller.
// When query_bytes_out is non-null, only the workspace-size query runs (nothing is
// launched); the result is written there and the outputs/workspace are not touched.
template <typename KeysInItT, typename KeysOutItItT, typename ValuesInItItT,
          typename ValuesOutItItT, typename SegmentSizesT, typename KArgT, typename NumSegsT,
          typename RequirementsT>
cudaError_t CUBBatchedTopKInvoke(KeysInItT d_keys_in, KeysOutItItT d_keys_out,
                                 ValuesInItItT d_values_in, ValuesOutItItT d_values_out,
                                 SegmentSizesT segment_sizes, KArgT k_arg, NumSegsT num_segs,
                                 RequirementsT requirements,
                                 Optional<TensorView>& maybe_workspace_buffer,
                                 size_t* query_bytes_out, cudaStream_t stream) {
  auto env = cuda::std::execution::env{requirements, cuda::stream_ref{stream}};

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
}

// Bottom of the dispatch chain: builds the two arguments whose types embed the compile-time
// segment-size bound (k_arg and segment_sizes) and invokes. Not API-specific — the API's
// identity lives entirely in the iterator types passed through. lengths is nullable:
// a device-side per-row lengths array selects the deferred segment sizes (both transform
// entries), nullptr means every segment spans the full max_len (plain top_k, which has no
// lengths). query_bytes_out non-null selects query-only, see CUBBatchedTopKInvoke.
template <int64_t MAX_LEN_BOUND, typename KeysInItT, typename KeysOutItItT, typename ValuesInItItT,
          typename ValuesOutItItT, typename RequirementsT>
cudaError_t CUBBatchedTopKDispatchBounds(KeysInItT d_keys_in, KeysOutItItT d_keys_out,
                                         ValuesInItItT d_values_in, ValuesOutItItT d_values_out,
                                         const int32_t* lengths,
                                         Optional<TensorView>& maybe_workspace_buffer,
                                         int64_t num_rows, int64_t max_len, int64_t top_k,
                                         RequirementsT requirements, size_t* query_bytes_out,
                                         cudaStream_t stream) {
  auto k_arg = cuda::args::immediate{top_k, cuda::args::bounds<int64_t{1}, MAX_LEN_BOUND>()};
  auto num_segs = cuda::args::immediate{num_rows};

  if (lengths != nullptr) {
    // Per-row segment sizes, read on device in stream order. The second, *runtime* bound is
    // a perf lever, not decoration: the host cannot read the device-side lengths when sizing
    // the launch, so without it CUB would size the cluster launch from the static bound-tier
    // ceiling, failing single-CTA eligibility and forcing the wide multi-CTA path for every
    // segment. Passing max_len as the runtime ceiling keeps small-row launches on the cheap
    // single-CTA shape.
    // The lower bound spans the full int32 range so no lengths value can violate the bounds
    // contract (out-of-bounds values are UB): under a negative statically-known lower bound,
    // CUB clamps any negative runtime size to an empty segment (size 0), and a zero-length
    // row is a valid empty segment — CUB selects nothing for it, so with the caller's -1
    // prefill the whole output row reads as padding. The lower bound plays no role in launch
    // sizing (only the upper bound does), so this costs nothing.
    constexpr int32_t k_lengths_floor = cuda::std::numeric_limits<int32_t>::min();
    const auto segment_sizes = cuda::args::deferred_sequence{
        lengths, cuda::args::bounds<k_lengths_floor, int32_t{MAX_LEN_BOUND}>(),
        cuda::args::bounds(k_lengths_floor, static_cast<int32_t>(max_len))};

    return CUBBatchedTopKInvoke(d_keys_in, d_keys_out, d_values_in, d_values_out, segment_sizes,
                                k_arg, num_segs, requirements, maybe_workspace_buffer,
                                query_bytes_out, stream);
  }

  // Uniform full-width rows: every segment is exactly max_len items, known on the host, so
  // the size is an immediate argument (no device read; CUB sizes the launch from the exact
  // value).
  const auto segment_sizes =
      cuda::args::immediate{max_len, cuda::args::bounds<int64_t{1}, MAX_LEN_BOUND>()};
  return CUBBatchedTopKInvoke(d_keys_in, d_keys_out, d_values_in, d_values_out, segment_sizes,
                              k_arg, num_segs, requirements, maybe_workspace_buffer,
                              query_bytes_out, stream);
}

// Fans the runtime tie_break flag out into its compile-time requirement configuration and
// picks the compile-time segment-size bound tier from the runtime max_len. Each require(...)
// call has a distinct type, so the three calls cannot share one requirements variable; the
// generic run_bound lambda factors the otherwise-identical bound branch.
template <typename KeysInItT, typename KeysOutItItT, typename ValuesInItItT,
          typename ValuesOutItItT>
cudaError_t CUBBatchedTopKDispatch(KeysInItT d_keys_in, KeysOutItItT d_keys_out,
                                   ValuesInItItT d_values_in, ValuesOutItItT d_values_out,
                                   const int32_t* lengths,
                                   Optional<TensorView>& maybe_workspace_buffer, int64_t num_rows,
                                   int64_t max_len, int64_t top_k, int64_t tie_break,
                                   size_t* query_bytes_out, cudaStream_t stream) {
  namespace exec = cuda::execution;

  auto run_bound = [&](auto requirements) {
    // CUB picks its backend from the compile-time bound: up to 8192 it can use the
    // single-block backend, which runs on any architecture. Anything larger needs the cluster
    // backend and therefore SM90+, so without this tier pre-SM90 GPUs couldn't run cub_topk
    // at all. 8192 is the largest bound the single-block backend accepts for our key/value
    // types.
    if (max_len <= 8192) {
      return CUBBatchedTopKDispatchBounds<int64_t{8192}>(
          d_keys_in, d_keys_out, d_values_in, d_values_out, lengths, maybe_workspace_buffer,
          num_rows, max_len, top_k, requirements, query_bytes_out, stream);
    } else {
      // Cluster backend, SM90+ only. On an older device this doesn't crash or fall back — the
      // CUB dispatch notices at runtime and returns cudaErrorNotSupported (see the
      // CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT define at the top of this file), which the
      // entry point turns into an exception. The Python dispatcher is expected to route these
      // calls to the radix backend instead, so reaching this branch pre-SM90 means someone
      // forced the CUB backend explicitly.
      return CUBBatchedTopKDispatchBounds<CUB_TOPK_MAX_LEN>(
          d_keys_in, d_keys_out, d_values_in, d_values_out, lengths, maybe_workspace_buffer,
          num_rows, max_len, top_k, requirements, query_bytes_out, stream);
    }
  };

  if (tie_break == 1) {
    return run_bound(exec::require(exec::determinism::gpu_to_gpu,
                                   exec::tie_break::prefer_smaller_index,
                                   exec::output_ordering::unsorted));
  } else if (tie_break == 2) {
    return run_bound(exec::require(exec::determinism::gpu_to_gpu,
                                   exec::tie_break::prefer_larger_index,
                                   exec::output_ordering::unsorted));
  } else {
    return run_bound(exec::require(exec::determinism::not_guaranteed, exec::tie_break::unspecified,
                                   exec::output_ordering::unsorted));
  }
}

// Top of the variable-length transform chain, shared by the page-table and ragged entries
// (their identity
// lives in the values-out iterator they pass): builds the bound-independent data iterators
// and picks the keys-in shape. d_keys_in[i] yields a pointer to row i's selection window in
// the dense (num_rows, max_len) input; the common whole-row case keeps the plain strided
// iterator (pure pointer arithmetic), only windowed calls pay the per-row device read.
template <typename DType, typename ValuesOutItItT>
cudaError_t CUBBatchedTopKVarLenTransform(const DType* input, int64_t row_stride,
                                          const int32_t* row_starts, ValuesOutItItT d_values_out,
                                          const int32_t* lengths,
                                          Optional<TensorView>& maybe_workspace_buffer,
                                          int64_t num_rows, int64_t max_len, int64_t top_k,
                                          int64_t tie_break, size_t* query_bytes_out,
                                          cudaStream_t stream) {
  // The keys (score values) are not returned by the transform API: every segment gets the
  // same discard iterator (the outer constant level satisfies the iterator-of-iterators
  // contract; a bare discard_iterator's reference is not an iterator and does not compile).
  auto d_keys_out = cuda::make_constant_iterator(cuda::discard_iterator{});
  // The "values" carried alongside each key are the per-segment item indices [0, max_len),
  // synthesized by a counting iterator; d_values_out then receives the top-k source indices.
  auto d_values_in = cuda::make_constant_iterator(cuda::make_counting_iterator(int32_t{0}));

  if (row_starts != nullptr) {
    auto d_keys_in =
        cuda::make_transform_iterator(cuda::make_counting_iterator(int64_t{0}),
                                      CUBMakeRowIn<DType>{input, row_stride, row_starts});
    return CUBBatchedTopKDispatch(d_keys_in, d_keys_out, d_values_in, d_values_out, lengths,
                                  maybe_workspace_buffer, num_rows, max_len, top_k, tie_break,
                                  query_bytes_out, stream);
  }
  auto d_keys_in = cuda::make_strided_iterator(cuda::make_counting_iterator(input), row_stride);
  return CUBBatchedTopKDispatch(d_keys_in, d_keys_out, d_values_in, d_values_out, lengths,
                                maybe_workspace_buffer, num_rows, max_len, top_k, tie_break,
                                query_bytes_out, stream);
}

// Widening store for the plain top_k indices: the API's contract is int64 (torch.topk
// drop-in), but the value pipeline stays int32 end-to-end inside CUB — only the final
// store casts. Carrying int64 values through the selection instead would double the value
// payload in registers/shared memory and could shrink the single-block tier's bound.
struct CUBCastIndexToInt64 {
  __device__ int64_t operator()(int32_t idx) const { return idx; }
};

// Top of the plain batched top-k chain: full-width rows (no lengths window, no row_starts) with
// both results returned — real strided writers for keys (the scores) and values (the
// indices) instead of the transform entries' discard/translate stacks. The cast is
// row-independent, so no per-row maker functor: the flat casting iterator composes with
// counting/strided exactly like a raw pointer.
template <typename DType>
cudaError_t CUBBatchedTopK(const DType* input, int64_t row_stride, DType* output_values,
                           int64_t* output_indices, Optional<TensorView>& maybe_workspace_buffer,
                           int64_t num_rows, int64_t max_len, int64_t top_k, int64_t tie_break,
                           size_t* query_bytes_out, cudaStream_t stream) {
  auto d_keys_in = cuda::make_strided_iterator(cuda::make_counting_iterator(input), row_stride);
  auto d_keys_out = cuda::make_strided_iterator(cuda::make_counting_iterator(output_values), top_k);
  // The "values" carried alongside each key are the per-segment item indices [0, max_len),
  // synthesized by a counting iterator; d_values_out then receives the top-k source indices.
  auto d_values_in = cuda::make_constant_iterator(cuda::make_counting_iterator(int32_t{0}));
  auto flat_indices_out =
      cuda::make_transform_output_iterator(output_indices, CUBCastIndexToInt64{});
  auto d_values_out =
      cuda::make_strided_iterator(cuda::make_counting_iterator(flat_indices_out), top_k);
  return CUBBatchedTopKDispatch(d_keys_in, d_keys_out, d_values_in, d_values_out,
                                /*lengths=*/nullptr, maybe_workspace_buffer, num_rows, max_len,
                                top_k, tie_break, query_bytes_out, stream);
}

// Input-side validation shared by every CUB entry and workspace-size query.
void CheckCUBTopKInput(const TensorView& input, int64_t top_k, int64_t tie_break) {
  // Rows only need to be individually contiguous: the row pitch is threaded through as
  // input.stride(0), so strided views (e.g. scores[:, :cur_len] of a wider buffer) work
  // without a .contiguous() copy.
  CHECK_CUDA(input);
  CHECK_LAST_DIM_CONTIGUOUS(input);
  CHECK_DIM(2, input);  // input: (batch_size, d)
  TVM_FFI_ICHECK(tie_break >= 0 && tie_break <= 2)
      << "Invalid tie_break mode " << tie_break
      << ", expected 0 (none), 1 (prefer small indices), or 2 (prefer large indices)";

  const int64_t max_len = input.size(1);
  TVM_FFI_ICHECK(top_k > 0 && top_k <= max_len)
      << "cub_topk requires 0 < top_k <= d, got top_k=" << top_k << ", d=" << max_len;
  TVM_FFI_ICHECK(max_len <= CUB_TOPK_MAX_LEN)
      << "cub_topk supports d <= " << CUB_TOPK_MAX_LEN << ", got d=" << max_len;
}

// Validation shared by the transform entries and their workspace-size queries.
void CheckCUBTopKArgs(const TensorView& input, const TensorView& lengths, int64_t top_k,
                      int64_t tie_break) {
  CheckCUBTopKInput(input, top_k, tie_break);
  CHECK_INPUT(lengths);
  CHECK_DIM(1, lengths);  // lengths: (batch_size,)
  CHECK_INPUT_TYPE(lengths, dl_int32);
  TVM_FFI_ICHECK(lengths.size(0) == input.size(0))
      << "cub_topk lengths must have one entry per row: expected " << input.size(0) << ", got "
      << lengths.size(0);
}

}  // namespace

// CUB-backed fused top-k + page-table transform. Supports the packed-layout arguments —
// row_to_batch (score row i gathers through table row row_to_batch[i]), row_starts (shifts
// the per-row read window), and page_table_row_starts (base offset into the table row, with
// row_starts as its fallback). For each row i, the top-k is selected over
// input[i, start:start+lengths[i]] and each winning window-local index idx is written as
//   src_page_table[i, idx / page_size] * page_size + idx % page_size
// with idx itself optionally duplicated into output_raw_indices. The translation happens
// inside CUB's own kernel via the output iterators — no epilogue launch. Rows with
// lengths[i] < top_k leave both output tails untouched; a post-selection tail kernel fills
// those suffixes with -1 (matching the native kernels, which write -1 in-kernel).
void cub_topk_page_table_transform(
    TensorView input, TensorView output_page_table, TensorView src_page_table, TensorView lengths,
    Optional<TensorView> maybe_output_raw_indices, Optional<TensorView> maybe_workspace_buffer,
    int64_t top_k, int64_t tie_break, int64_t page_size, Optional<TensorView> maybe_row_to_batch,
    Optional<TensorView> maybe_row_starts, Optional<TensorView> maybe_page_table_row_starts) {
  CheckPageTableTransformArgs(input, output_page_table, src_page_table, lengths, maybe_row_to_batch,
                              maybe_row_starts, maybe_page_table_row_starts,
                              maybe_output_raw_indices, top_k, page_size);
  // CUB-specific constraints; everything shared with the radix launcher lives in
  // CheckPageTableTransformArgs.
  TVM_FFI_ICHECK(tie_break >= 0 && tie_break <= 2)
      << "Invalid tie_break mode " << tie_break
      << ", expected 0 (none), 1 (prefer small indices), or 2 (prefer large indices)";
  TVM_FFI_ICHECK(top_k <= input.size(1))
      << "cub_topk requires top_k <= d, got top_k=" << top_k << ", d=" << input.size(1);
  TVM_FFI_ICHECK(input.size(1) <= CUB_TOPK_MAX_LEN)
      << "cub_topk supports d <= " << CUB_TOPK_MAX_LEN << ", got d=" << input.size(1);

  const auto* lengths_ptr = static_cast<const int32_t*>(lengths.data_ptr());
  const auto* row_to_batch_ptr =
      maybe_row_to_batch.has_value()
          ? static_cast<const int32_t*>(maybe_row_to_batch.value().data_ptr())
          : nullptr;
  const auto* row_starts_ptr =
      maybe_row_starts.has_value()
          ? static_cast<const int32_t*>(maybe_row_starts.value().data_ptr())
          : nullptr;
  const auto* page_table_row_starts_ptr =
      maybe_page_table_row_starts.has_value()
          ? static_cast<const int32_t*>(maybe_page_table_row_starts.value().data_ptr())
          : nullptr;

  auto* output_raw_indices =
      maybe_output_raw_indices.has_value()
          ? static_cast<int32_t*>(maybe_output_raw_indices.value().data_ptr())
          : nullptr;

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
      status = CUBBatchedTopKVarLenTransform(
          static_cast<const c_type*>(input.data_ptr()), input.stride(0), row_starts_ptr,
          d_values_out, lengths_ptr, maybe_workspace_buffer, num_rows, max_len, top_k, tie_break,
          /*query_bytes_out=*/nullptr, stream);
      return true;
    });
  };
  if (output_raw_indices != nullptr) {
    run(CUBMakePageTableRowOut<true>{out_ptr, output_raw_indices, src_ptr, row_to_batch_ptr,
                                     row_starts_ptr, page_table_row_starts_ptr, page_stride,
                                     page_bits, top_k});
  } else {
    run(CUBMakePageTableRowOut<false>{out_ptr, nullptr, src_ptr, row_to_batch_ptr, row_starts_ptr,
                                      page_table_row_starts_ptr, page_stride, page_bits, top_k});
  }

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk_page_table_transform failed with error code " << cudaGetErrorString(status);
  status = CUBFillTopKTails(out_ptr, output_raw_indices, lengths_ptr, num_rows, top_k, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk_page_table_transform tail fill failed with error code "
      << cudaGetErrorString(status);
}

// Workspace query for the transform variant. Must instantiate the same dispatch (same
// iterator types) as the run so the byte count matches the kernel that will execute — the
// output mode changes the output iterator type (with_raw_indices) and a row_starts window
// changes the input iterator type (with_row_starts), so the caller states both.
int64_t cub_topk_page_table_transform_workspace_size(TensorView input, TensorView lengths,
                                                     int64_t top_k, int64_t tie_break,
                                                     bool with_raw_indices, bool with_row_starts) {
  CheckCUBTopKArgs(input, lengths, top_k, tie_break);
  const auto* lengths_ptr = static_cast<const int32_t*>(lengths.data_ptr());

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
  // A non-null row_starts selects the CUBMakeRowIn input iterator, matching the run's
  // instantiation; lengths_ptr is a stand-in base that is never dereferenced.
  const int32_t* row_starts_stub = with_row_starts ? lengths_ptr : nullptr;
  auto query = [&](auto row_out_maker) {
    auto d_values_out =
        cuda::make_transform_iterator(cuda::make_counting_iterator(int64_t{0}), row_out_maker);
    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(input.dtype(), c_type, [&] {
      status = CUBBatchedTopKVarLenTransform(static_cast<const c_type*>(input.data_ptr()),
                                             input.stride(0), row_starts_stub, d_values_out,
                                             lengths_ptr, no_workspace, num_rows, max_len, top_k,
                                             tie_break, &workspace_bytes, stream);
      return true;
    });
  };
  if (with_raw_indices) {
    query(CUBMakePageTableRowOut<true>{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0,
                                       top_k});
  } else {
    query(CUBMakePageTableRowOut<false>{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0,
                                        top_k});
  }

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk_page_table_transform workspace-size query failed with error code "
      << cudaGetErrorString(status);
  return static_cast<int64_t>(workspace_bytes);
}

// CUB-backed fused top-k + ragged index transform. For each row i, the top-k is selected
// over input[i, 0:lengths[i]] (shifted right by row_starts[i] when given) and each winning
// window-local index idx is written as idx + offsets[i] — the shift into the caller's
// global coordinates happens inside CUB's own kernel via the output iterators, no epilogue
// launch. row_starts moves the read window only and does not add into the output (matching
// the native ragged kernel). Rows with lengths[i] < top_k leave the output tail untouched;
// a post-selection tail kernel fills that suffix with -1 (matching the native kernels,
// which write -1 in-kernel).
void cub_topk_ragged_transform(TensorView input, TensorView output_indices, TensorView offsets,
                               TensorView lengths, Optional<TensorView> maybe_workspace_buffer,
                               int64_t top_k, int64_t tie_break,
                               Optional<TensorView> maybe_row_starts) {
  CheckCUBTopKArgs(input, lengths, top_k, tie_break);
  const int64_t num_rows = input.size(0);
  const int64_t max_len = input.size(1);
  TVM_FFI_ICHECK_GE(input.stride(0), max_len) << "input rows must not overlap";

  CHECK_INPUT_AND_TYPE(output_indices, dl_int32);
  CHECK_DEVICE(output_indices, input);
  CHECK_DIM(2, output_indices);  // output_indices: (num_rows, top_k)
  TVM_FFI_ICHECK_EQ(output_indices.size(0), num_rows)
      << "output_indices must have shape (num_rows, top_k)";
  TVM_FFI_ICHECK_EQ(output_indices.size(1), top_k)
      << "output_indices must have shape (num_rows, top_k)";

  CHECK_INPUT_AND_TYPE(offsets, dl_int32);
  CHECK_DEVICE(offsets, input);
  CHECK_DIM(1, offsets);  // offsets: (num_rows,)
  TVM_FFI_ICHECK_EQ(offsets.size(0), num_rows) << "offsets must have one entry per row: expected "
                                               << num_rows << ", got " << offsets.size(0);

  if (maybe_row_starts.has_value()) {
    const auto& row_starts = maybe_row_starts.value();
    CHECK_INPUT_AND_TYPE(row_starts, dl_int32);
    CHECK_DEVICE(row_starts, input);
    CHECK_DIM(1, row_starts);  // row_starts: (num_rows,)
    TVM_FFI_ICHECK_EQ(row_starts.size(0), num_rows)
        << "row_starts must have one entry per row: expected " << num_rows << ", got "
        << row_starts.size(0);
  }

  if (num_rows == 0) {
    return;
  }

  const auto* lengths_ptr = static_cast<const int32_t*>(lengths.data_ptr());
  const auto* offsets_ptr = static_cast<const int32_t*>(offsets.data_ptr());
  const auto* row_starts_ptr =
      maybe_row_starts.has_value()
          ? static_cast<const int32_t*>(maybe_row_starts.value().data_ptr())
          : nullptr;
  auto* out_ptr = static_cast<int32_t*>(output_indices.data_ptr());

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  auto d_values_out = cuda::make_transform_iterator(
      cuda::make_counting_iterator(int64_t{0}), CUBMakeRaggedRowOut{out_ptr, offsets_ptr, top_k});
  cudaError_t status = cudaErrorInvalidValue;
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(input.dtype(), c_type, [&] {
    status = CUBBatchedTopKVarLenTransform(
        static_cast<const c_type*>(input.data_ptr()), input.stride(0), row_starts_ptr, d_values_out,
        lengths_ptr, maybe_workspace_buffer, num_rows, max_len, top_k, tie_break,
        /*query_bytes_out=*/nullptr, stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk_ragged_transform failed with error code " << cudaGetErrorString(status);
  status = CUBFillTopKTails(out_ptr, /*output_raw_indices=*/nullptr, lengths_ptr, num_rows, top_k,
                            stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk_ragged_transform tail fill failed with error code "
      << cudaGetErrorString(status);
}

// Workspace query for the ragged variant. Must instantiate the same dispatch (same iterator
// types) as the run so the byte count matches the kernel that will execute; the ragged API
// has a single output mode, but a row_starts window changes the input iterator type, so the
// caller states it via with_row_starts.
int64_t cub_topk_ragged_transform_workspace_size(TensorView input, TensorView lengths,
                                                 int64_t top_k, int64_t tie_break,
                                                 bool with_row_starts) {
  CheckCUBTopKArgs(input, lengths, top_k, tie_break);
  const auto* lengths_ptr = static_cast<const int32_t*>(lengths.data_ptr());

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
  // A non-null row_starts selects the CUBMakeRowIn input iterator, matching the run's
  // instantiation; lengths_ptr is a stand-in base that is never dereferenced.
  const int32_t* row_starts_stub = with_row_starts ? lengths_ptr : nullptr;
  auto d_values_out = cuda::make_transform_iterator(cuda::make_counting_iterator(int64_t{0}),
                                                    CUBMakeRaggedRowOut{nullptr, nullptr, top_k});
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(input.dtype(), c_type, [&] {
    status = CUBBatchedTopKVarLenTransform(static_cast<const c_type*>(input.data_ptr()),
                                           input.stride(0), row_starts_stub, d_values_out,
                                           lengths_ptr, no_workspace, num_rows, max_len, top_k,
                                           tie_break, &workspace_bytes, stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk_ragged_transform workspace-size query failed with error code "
      << cudaGetErrorString(status);
  return static_cast<int64_t>(workspace_bytes);
}

// CUB-backed plain batched top-k (torch.topk-style): for each row i of the dense
// (num_rows, d) input, the top_k largest (value, index) pairs are written to
// output_values[i] / output_indices[i], unsorted. Every row is full width — there is no
// lengths window and no -1 padding; every output slot is written. Indices are written as
// int64 directly (torch.topk drop-in contract) via a widening output iterator — no
// conversion kernel at the Python boundary.
void cub_topk(TensorView input, TensorView output_indices, TensorView output_values,
              Optional<TensorView> maybe_workspace_buffer, int64_t top_k, int64_t tie_break) {
  CheckCUBTopKInput(input, top_k, tie_break);
  const int64_t num_rows = input.size(0);
  const int64_t max_len = input.size(1);
  TVM_FFI_ICHECK_GE(input.stride(0), max_len) << "input rows must not overlap";

  CHECK_INPUT_AND_TYPE(output_indices, dl_int64);
  CHECK_DEVICE(output_indices, input);
  CHECK_DIM(2, output_indices);  // output_indices: (num_rows, top_k)
  TVM_FFI_ICHECK_EQ(output_indices.size(0), num_rows)
      << "output_indices must have shape (num_rows, top_k)";
  TVM_FFI_ICHECK_EQ(output_indices.size(1), top_k)
      << "output_indices must have shape (num_rows, top_k)";

  CHECK_INPUT(output_values);
  CHECK_DEVICE(output_values, input);
  CHECK_DIM(2, output_values);  // output_values: (num_rows, top_k)
  CHECK_SHAPE(output_values, output_indices);
  TVM_FFI_ICHECK(input.dtype() == output_values.dtype())
      << "output_values must have the same dtype as input";

  if (num_rows == 0) {
    return;
  }

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  auto* indices_ptr = static_cast<int64_t*>(output_indices.data_ptr());
  cudaError_t status = cudaErrorInvalidValue;
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(input.dtype(), c_type, [&] {
    status = CUBBatchedTopK(static_cast<const c_type*>(input.data_ptr()), input.stride(0),
                            static_cast<c_type*>(output_values.data_ptr()), indices_ptr,
                            maybe_workspace_buffer, num_rows, max_len, top_k, tie_break,
                            /*query_bytes_out=*/nullptr, stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk failed with error code " << cudaGetErrorString(status);
}

// Workspace query for the plain variant. Must instantiate the same dispatch (same iterator
// types) as the run so the byte count matches the kernel that will execute.
int64_t cub_topk_workspace_size(TensorView input, int64_t top_k, int64_t tie_break) {
  CheckCUBTopKInput(input, top_k, tie_break);

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
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(input.dtype(), c_type, [&] {
    status =
        CUBBatchedTopK(static_cast<const c_type*>(input.data_ptr()), input.stride(0),
                       static_cast<c_type*>(nullptr), static_cast<int64_t*>(nullptr), no_workspace,
                       num_rows, max_len, top_k, tie_break, &workspace_bytes, stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk workspace-size query failed with error code " << cudaGetErrorString(status);
  return static_cast<int64_t>(workspace_bytes);
}
