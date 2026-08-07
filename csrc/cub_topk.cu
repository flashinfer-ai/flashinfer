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

#include <cub/device/device_batched_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/__execution/env.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace {

// cub::DeviceBatchedTopK requires a compile-time upper bound on the segment size, so the
// runtime max_len is dispatched over a small ladder of bounds (see
// CUBBatchedTopKDispatchBound). Every tier multiplies kernel instantiations (tiers x dtypes x
// requirement configs x segment-size argument forms), so the ladder is kept minimal: one tier
// per capability class.
//
// 2^21 is DeviceBatchedTopK's own per-segment limit — it static_asserts on anything larger
// ("larger segments are future work" per the CUB docs). So this is both our top tier and the
// most this backend can ever handle; the Python dispatcher has to send longer rows to the
// radix backend.
constexpr int64_t CUB_TOPK_MAX_LEN = int64_t{1} << 21;

// When query_bytes_out is non-null, only the temp-storage size query runs (nothing is
// launched); the result is written there and the outputs/workspace are not touched.
template <int64_t MAX_LEN_BOUND, typename DType, typename RequirementsT>
cudaError_t CUBBatchedTopKRun(const DType* input, int64_t row_stride, int32_t* output_indices,
                              DType* output_values, const int32_t* lengths,
                              Optional<TensorView>& maybe_temp_storage, int64_t num_rows,
                              int64_t max_len, int64_t top_k, const RequirementsT& requirements,
                              size_t* query_bytes_out, cudaStream_t stream) {
  // Per-segment iterators over the dense (num_rows, max_len) input and (num_rows, top_k)
  // outputs: d_keys_in[i] yields a pointer to row i.
  auto d_keys_in = cuda::make_strided_iterator(cuda::make_counting_iterator(input), row_stride);
  auto d_keys_out =
      cuda::make_strided_iterator(cuda::make_counting_iterator(output_values), top_k);
  // The "values" carried alongside each key are the per-segment item indices [0, max_len),
  // synthesized by a counting iterator; d_values_out then receives the top-k source indices.
  auto d_values_in = cuda::make_constant_iterator(cuda::make_counting_iterator(int32_t{0}));
  auto d_values_out =
      cuda::make_strided_iterator(cuda::make_counting_iterator(output_indices), top_k);

  auto k_arg = cuda::args::immediate{top_k, cuda::args::bounds<int64_t{1}, MAX_LEN_BOUND>()};
  auto num_segs = cuda::args::immediate{num_rows};

  auto env = cuda::std::execution::env{requirements, cuda::stream_ref{stream}};

  // The uniform and per-row-lengths paths differ only in the segment_sizes argument (distinct
  // types); the two-phase temp-storage flow is shared through this generic lambda.
  auto run_with = [&](auto segment_sizes) -> cudaError_t {
    size_t temp_storage_bytes = 0;
    if (const auto error =
            cub::DeviceBatchedTopK::MaxPairs(nullptr, temp_storage_bytes, d_keys_in, d_keys_out,
                                             d_values_in, d_values_out, segment_sizes, k_arg,
                                             num_segs, env)) {
      return error;
    }
    if (query_bytes_out != nullptr) {
      *query_bytes_out = temp_storage_bytes;
      return cudaSuccess;
    }

    void* d_temp_storage = nullptr;
    bool owned = false;
    if (maybe_temp_storage.has_value()) {
      const auto& workspace = maybe_temp_storage.value();
      const size_t workspace_bytes =
          static_cast<size_t>(workspace.numel()) * get_element_size(workspace);
      TVM_FFI_ICHECK(workspace_bytes >= temp_storage_bytes)
          << "cub_topk workspace too small: need " << temp_storage_bytes << " bytes, have "
          << workspace_bytes;
      d_temp_storage = workspace.data_ptr();
    } else {
      if (const auto error = cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream)) {
        return error;
      }
      owned = true;
    }

    // No early return below this point: the owned temp storage must be freed on every path.
    cudaError_t status = cub::DeviceBatchedTopK::MaxPairs(
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
        segment_sizes, k_arg, num_segs, env);

    if (owned) {
      if (const auto free_error = cudaFreeAsync(d_temp_storage, stream)) {
        // Prefer the MaxPairs error over the free error when both fail.
        return status == cudaSuccess ? free_error : status;
      }
    }
    return status;
  };

  if (lengths != nullptr) {
    // Per-row segment sizes, read on device in stream order. The second, *runtime* bound is a
    // perf lever, not decoration: the host cannot read the device-side lengths when sizing the
    // launch, so without it CUB would size the cluster launch from the static MAX_LEN_BOUND
    // ceiling, failing single-CTA eligibility and forcing the wide multi-CTA path for every
    // segment. Passing max_len as the runtime ceiling restores the same launch shape as the
    // uniform path.
    return run_with(cuda::args::deferred_sequence{
        lengths, cuda::args::bounds<int32_t{1}, int32_t{MAX_LEN_BOUND}>(),
        cuda::args::bounds(int32_t{1}, static_cast<int32_t>(max_len))});
  }
  return run_with(
      cuda::args::immediate{max_len, cuda::args::bounds<int64_t{1}, MAX_LEN_BOUND>()});
}

template <typename DType, typename RequirementsT>
cudaError_t CUBBatchedTopKDispatchBound(const DType* input, int64_t row_stride,
                                        int32_t* output_indices, DType* output_values,
                                        const int32_t* lengths,
                                        Optional<TensorView>& maybe_temp_storage,
                                        int64_t num_rows, int64_t max_len, int64_t top_k,
                                        const RequirementsT& requirements,
                                        size_t* query_bytes_out, cudaStream_t stream) {
  // CUB picks its backend from the compile-time bound: up to 8192 it can use the single-block
  // backend, which runs on any architecture. Anything larger needs the cluster backend and
  // therefore SM90+, so without this tier pre-SM90 GPUs couldn't run cub_topk at all. 8192 is
  // the largest bound the single-block backend accepts for our key/value types.
  if (max_len <= 8192) {
    return CUBBatchedTopKRun<int64_t{8192}>(input, row_stride, output_indices, output_values,
                                            lengths, maybe_temp_storage, num_rows, max_len,
                                            top_k, requirements, query_bytes_out, stream);
  } else {
    // Cluster backend, SM90+ only. On an older device this doesn't crash or fall back — the
    // CUB dispatch notices at runtime and returns cudaErrorNotSupported (see the
    // CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT define at the top of this file), which
    // cub_topk() turns into an exception. The Python dispatcher is expected to route these
    // calls to the radix backend instead, so reaching this branch pre-SM90 means someone
    // forced the CUB backend explicitly.
    return CUBBatchedTopKRun<CUB_TOPK_MAX_LEN>(input, row_stride, output_indices, output_values,
                                               lengths, maybe_temp_storage, num_rows, max_len,
                                               top_k, requirements, query_bytes_out, stream);
  }
}

template <typename DType>
cudaError_t CUBBatchedTopKDispatch(const DType* input, int64_t row_stride,
                                   int32_t* output_indices, DType* output_values,
                                   const int32_t* lengths,
                                   Optional<TensorView>& maybe_temp_storage, int64_t num_rows,
                                   int64_t max_len, int64_t top_k, int64_t tie_break,
                                   size_t* query_bytes_out, cudaStream_t stream) {
  namespace exec = cuda::execution;
  // Each require(...) call has a distinct type (requirements are encoded at compile time), so
  // the runtime flag must fan out into separate branches; the generic lambda factors out the
  // otherwise-identical call.
  auto run = [&](auto requirements) {
    return CUBBatchedTopKDispatchBound(input, row_stride, output_indices, output_values,
                                       lengths, maybe_temp_storage, num_rows, max_len, top_k,
                                       requirements, query_bytes_out, stream);
  };

  if (tie_break == 1) {
    return run(exec::require(exec::determinism::gpu_to_gpu,
                             exec::tie_break::prefer_smaller_index,
                             exec::output_ordering::unsorted));
  } else if (tie_break == 2) {
    return run(exec::require(exec::determinism::gpu_to_gpu,
                             exec::tie_break::prefer_larger_index,
                             exec::output_ordering::unsorted));
  } else {
    return run(exec::require(exec::determinism::not_guaranteed, exec::tie_break::unspecified,
                             exec::output_ordering::unsorted));
  }
}

// Validation shared by cub_topk and cub_topk_workspace_size. Returns the lengths pointer
// (nullptr when absent).
const int32_t* CheckCUBTopKArgs(const TensorView& input, const Optional<TensorView>& maybe_lengths,
                                int64_t top_k, int64_t tie_break) {
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

  if (!maybe_lengths.has_value()) {
    return nullptr;
  }
  const auto& lengths = maybe_lengths.value();
  CHECK_INPUT(lengths);
  CHECK_DIM(1, lengths);  // lengths: (batch_size,)
  CHECK_INPUT_TYPE(lengths, dl_int32);
  TVM_FFI_ICHECK(lengths.size(0) == num_rows)
      << "cub_topk lengths must have one entry per row: expected " << num_rows << ", got "
      << lengths.size(0);
  return static_cast<const int32_t*>(lengths.data_ptr());
}

}  // namespace

// CUB-backed batched top-k over the rows of a dense (num_rows, max_len) tensor.
//   maybe_lengths: optional (num_rows,) int32 per-row valid sizes; row i selects its top-k over
//     input[i, 0:lengths[i]]. Absent => every row uses the full width max_len. Rows with
//     lengths[i] < top_k return all lengths[i] elements, with the remaining output_indices
//     padded to -1 (output_values at padded positions are unspecified).
void cub_topk(TensorView input, TensorView output_indices, TensorView output_values,
              Optional<TensorView> maybe_lengths, Optional<TensorView> maybe_temp_storage,
              int64_t top_k, int64_t tie_break) {
  const int32_t* lengths_ptr = CheckCUBTopKArgs(input, maybe_lengths, top_k, tie_break);
  CHECK_INPUT(output_indices);
  CHECK_INPUT(output_values);
  CHECK_DIM(2, output_indices);  // output_indices: (batch_size, top_k)
  CHECK_DIM(2, output_values);   // output_values: (batch_size, top_k)
  CHECK_INPUT_TYPE(output_indices, dl_int32);

  const int64_t num_rows = input.size(0);
  const int64_t max_len = input.size(1);
  if (num_rows == 0) {
    return;
  }

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  if (lengths_ptr != nullptr) {
    // For a row with lengths[i] < top_k, CUB clamps k to the segment size and writes only
    // lengths[i] pairs, leaving the tail of the output row untouched. Pre-fill the indices
    // with -1 so the padding is well-defined, matching the convention of FlashInfer's
    // lengths-bearing transform APIs. memset is byte-wise, but int32 -1 is 0xFFFFFFFF, so
    // filling every byte with 0xFF reads back as -1.
    TVM_FFI_ICHECK(cudaMemsetAsync(output_indices.data_ptr(), 0xFF,
                                   sizeof(int32_t) * num_rows * top_k, stream) == cudaSuccess)
        << "cub_topk output_indices pre-fill failed";
  }

  cudaError_t status;
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(input.dtype(), c_type, [&] {
    auto* input_ptr = static_cast<c_type*>(input.data_ptr());
    auto* indices_ptr = static_cast<int32_t*>(output_indices.data_ptr());
    auto* values_ptr = static_cast<c_type*>(output_values.data_ptr());
    status = CUBBatchedTopKDispatch(input_ptr, input.stride(0), indices_ptr, values_ptr,
                                    lengths_ptr, maybe_temp_storage, num_rows, max_len, top_k,
                                    tie_break, /*query_bytes_out=*/nullptr, stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk failed with error code " << cudaGetErrorString(status);
}

// Returns the temporary-storage bytes DeviceBatchedTopK needs for this problem shape and
// requirement configuration, so the caller can allocate a workspace once (outside CUDA graph
// capture) and pass it to every cub_topk call. Launches nothing.
int64_t cub_topk_workspace_size(TensorView input, Optional<TensorView> maybe_lengths,
                                int64_t top_k, int64_t tie_break) {
  const int32_t* lengths_ptr = CheckCUBTopKArgs(input, maybe_lengths, top_k, tie_break);

  const int64_t num_rows = input.size(0);
  const int64_t max_len = input.size(1);
  if (num_rows == 0) {
    return 0;
  }

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  auto stream = get_stream(input.device());

  Optional<TensorView> no_workspace;
  size_t temp_storage_bytes = 0;
  cudaError_t status;
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(input.dtype(), c_type, [&] {
    // The size query only inspects types and bounds; output pointers are never dereferenced.
    status = CUBBatchedTopKDispatch(static_cast<const c_type*>(input.data_ptr()),
                                    input.stride(0), static_cast<int32_t*>(nullptr),
                                    static_cast<c_type*>(nullptr), lengths_ptr, no_workspace,
                                    num_rows, max_len, top_k, tie_break, &temp_storage_bytes,
                                    stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk workspace-size query failed with error code " << cudaGetErrorString(status);
  return static_cast<int64_t>(temp_storage_bytes);
}
