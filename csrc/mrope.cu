/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 by FlashInfer team.
 */
#include <flashinfer/mrope.cuh>
#include <flashinfer/utils.cuh>
#include <limits>

#include "tvm_ffi_utils.h"
using namespace flashinfer;

// Conservative enclosing spans also protect metadata that lies in row padding.
struct MropeSpan {
  uintptr_t begin, end;
};

static MropeSpan mrope_span(TensorView t) {
  const auto begin = reinterpret_cast<uintptr_t>(t.data_ptr());
  uint64_t elements = 1;
  for (int i = 0; i < t.ndim(); ++i) {
    TVM_FFI_ICHECK(t.stride(i) >= 0) << "negative stride";
    if (t.size(i) == 0) return {begin, begin};
    const auto count = static_cast<uint64_t>(t.size(i) - 1);
    const auto stride = static_cast<uint64_t>(t.stride(i));
    TVM_FFI_ICHECK(
        count == 0 ||
        stride <= (static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - elements) / count)
        << "tensor span overflow";
    elements += count * stride;
  }
  const auto bytes = t.dtype().bits / 8;
  TVM_FFI_ICHECK(elements <= (std::numeric_limits<uintptr_t>::max() - begin) / bytes)
      << "tensor address overflow";
  return {begin, begin + elements * bytes};
}

static bool mrope_span_overlap(MropeSpan a, MropeSpan b) {
  return a.begin < a.end && b.begin < b.end && a.begin < b.end && b.begin < a.end;
}

static void apply_mrope_impl(TensorView q, TensorView k, TensorView positions, TensorView cache,
                             int64_t head_size, int64_t st, int64_t sh, int64_t sw,
                             bool axis_interleaved, bool validate_only) {
  for (auto t : {q, k, positions, cache}) {
    CHECK_CUDA(t);
    CHECK_DEVICE(q, t);
  }
  CHECK_DIM(2, q);
  CHECK_DIM(2, k);
  CHECK_DIM(2, positions);
  CHECK_DIM(2, cache);
  CHECK_SAME_DTYPE(q, k);
  TVM_FFI_ICHECK(q.dtype().lanes == 1 && q.dtype().bits == 16 &&
                 (q.dtype().code == kDLFloat || q.dtype().code == kDLBfloat))
      << "Q/K must be FP16 or BF16";
  TVM_FFI_ICHECK(positions.dtype().lanes == 1 && positions.dtype().code == kDLInt &&
                 (positions.dtype().bits == 32 || positions.dtype().bits == 64));
  TVM_FFI_ICHECK(
      cache.dtype() == q.dtype() ||
      (cache.dtype().code == kDLFloat && cache.dtype().bits == 32 && cache.dtype().lanes == 1));
  CHECK_CONTIGUOUS(cache);
  TVM_FFI_ICHECK_EQ(q.size(0), k.size(0));
  TVM_FFI_ICHECK(q.size(0) <= std::numeric_limits<int>::max());
  const auto d = head_size;
  const auto r = cache.size(1);
  TVM_FFI_ICHECK(d >= 2 && d <= 512 && d % 2 == 0) << "D must be even and in [2,512]";
  TVM_FFI_ICHECK(r >= 2 && r <= d && r % 2 == 0 && cache.size(0) > 0)
      << "R must be even and in [2,D]; cache must have rows";
  TVM_FFI_ICHECK(q.size(1) % d == 0 && k.size(1) % d == 0)
      << "row widths must be divisible by head_size";
  const auto hq = q.size(1) / d, hk = k.size(1) / d;
  TVM_FFI_ICHECK(hq > 0 && hq <= 1024 && hk > 0 && hk <= 1024)
      << "Q/K head counts must be in [1,1024]";
  TVM_FFI_ICHECK(positions.size(0) == 3 && positions.size(1) == q.size(0));
  for (auto t : {q, k}) {
    TVM_FFI_ICHECK(t.stride(1) == 1 && t.stride(0) >= t.size(1)) << "invalid Q/K row strides";
  }
  TVM_FFI_ICHECK(positions.stride(1) == 1 && positions.stride(0) >= positions.size(1))
      << "invalid position row strides";
  TVM_FFI_ICHECK(st >= 0 && sh >= 0 && sw >= 0 && st <= r / 2 && sh <= r / 2 && sw <= r / 2 &&
                 st + sh + sw == r / 2);
  if (axis_interleaved) {
    TVM_FFI_ICHECK(sh <= (r / 2 + 1) / 3 && sw <= (r / 2) / 3)
        << "interleaved axis sections do not fit the rotary dimension";
  }
  const auto qs = mrope_span(q), ks = mrope_span(k);
  const auto ps = mrope_span(positions), cs = mrope_span(cache);
  for (auto out : {qs, ks}) {
    TVM_FFI_ICHECK(!mrope_span_overlap(out, ps) && !mrope_span_overlap(out, cs))
        << "output aliases metadata";
  }
  if (mrope_span_overlap(qs, ks)) {
    // Equal-pitch disjoint row segments are safe even though their enclosing
    // spans overlap. Reject shifted rows, unequal pitches, and true overlap.
    TVM_FFI_ICHECK_EQ(q.stride(0), k.stride(0)) << "overlapping spans need equal row strides";
    const auto pitch = static_cast<uint64_t>(q.stride(0));
    TVM_FFI_ICHECK(pitch <= std::numeric_limits<uintptr_t>::max() / 2);
    const bool q_first = qs.begin <= ks.begin;
    const auto delta = q_first ? ks.begin - qs.begin : qs.begin - ks.begin;
    const uint64_t first_bytes = (q_first ? hq : hk) * d * 2;
    const uint64_t second_bytes = (q_first ? hk : hq) * d * 2;
    TVM_FFI_ICHECK(delta >= first_bytes && delta <= pitch * 2 && second_bytes <= pitch * 2 - delta)
        << "Q/K row segments overlap or cross rows";
  }
  if (validate_only || q.size(0) == 0) return;
  ffi::CUDADeviceGuard guard(q.device().device_id);
  const auto stream = get_stream(q.device());
  // SM120 measurements favor pair mapping for small eight-head batches and
  // wider prefill blocks in the middle range. Keep the original routing on
  // other architectures and beyond the measured crossover.
  bool pair_prefill = false, wide_prefill = false;
  if (d == 128 && hq == 8 && ((r == 128 && hk == 4) || (r == 64 && hk == 1)) && q.size(0) >= 96 &&
      q.size(0) <= 384) {
    const auto capability = GetCudaComputeCapability();
    if (capability.first == 12 && capability.second == 0) {
      pair_prefill = q.size(0) < (r == 128 ? 128 : 224);
      wide_prefill = !pair_prefill;
    }
  }
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q.dtype(), T, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(cache.dtype(), C, [&] {
      return DISPATCH_DLPACK_IDTYPE_TO_CTYPE(positions.dtype(), I, [&] {
        if (d == 128 && r == 128 && hq == 16 && hk == 8) {
          MropeToken128<T, C, I, 128, true><<<q.size(0), 128, 0, stream>>>(
              static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
              static_cast<const I*>(positions.data_ptr()), static_cast<const C*>(cache.data_ptr()),
              q.size(0), st, sh, sw, axis_interleaved, q.stride(0), k.stride(0),
              positions.stride(0));
          const auto status = cudaGetLastError();
          TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
          return true;
        }
        // Measured on SM86: fewer than eight Q heads retain the pair mapping.
        // Partial-rotary/smaller-head prefills also retain it below T=96.
        if (!pair_prefill && q.size(0) >= 32 && d == 128 && (r == 64 || r == 128) && hq >= 8 &&
            hq + hk <= 64 && (q.size(0) >= 96 || (r == 128 && hq + hk >= 16))) {
          return DISPATCH_BOOL(r == 128, Full, [&] {
            return DISPATCH_BOOL(wide_prefill, Wide, [&] {
              constexpr int threads = Wide ? 256 : 128;
              MropeTokenPrefill<T, C, I, Full ? 128 : 64, threads>
                  <<<q.size(0), threads, 0, stream>>>(
                      static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
                      static_cast<const I*>(positions.data_ptr()),
                      static_cast<const C*>(cache.data_ptr()), hq, hk, st, sh, sw, axis_interleaved,
                      q.stride(0), k.stride(0), positions.stride(0));
              const auto status = cudaGetLastError();
              TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
              return true;
            });
          });
        }
        // The generic path owns one rotary pair per thread and leaves the
        // non-rotary tail untouched. Strides are compile-time optional so the
        // original contiguous kernels retain their address calculation.
        constexpr int threads = 256;
        int pair_bits = 0;
        while ((1 << pair_bits) < r / 2) ++pair_bits;
        const bool dense = (1 << pair_bits) > d / 2;
        const int pitch = dense ? d / 2 : (1 << pair_bits);
        const auto blocks_y = ((hq + hk) * pitch + threads - 1) / threads;
        return DISPATCH_BOOL(dense, Dense, [&] {
          MropePairKernel<T, C, I, Dense, true><<<dim3(q.size(0), blocks_y), threads, 0, stream>>>(
              static_cast<const T*>(q.data_ptr()), static_cast<const T*>(k.data_ptr()),
              static_cast<T*>(q.data_ptr()), static_cast<T*>(k.data_ptr()),
              static_cast<const I*>(positions.data_ptr()), static_cast<const C*>(cache.data_ptr()),
              q.size(0), hq, hk, d, r, st, sh, sw, axis_interleaved, pair_bits, q.stride(0),
              k.stride(0), positions.stride(0));
          const auto status = cudaGetLastError();
          TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
          return true;
        });
      });
    });
  });
}

void apply_mrope(TensorView q, TensorView k, TensorView positions, TensorView cache,
                 int64_t head_size, int64_t st, int64_t sh, int64_t sw, bool axis_interleaved) {
  apply_mrope_impl(q, k, positions, cache, head_size, st, sh, sw, axis_interleaved, false);
}

void validate_mrope(TensorView q, TensorView k, TensorView positions, TensorView cache,
                    int64_t head_size, int64_t st, int64_t sh, int64_t sw, bool axis_interleaved) {
  apply_mrope_impl(q, k, positions, cache, head_size, st, sh, sw, axis_interleaved, true);
}
