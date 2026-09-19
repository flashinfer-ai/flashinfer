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
#include <flashinfer/attention/qsa_output_gate.cuh>
#include <utility>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

void qsa_output_gate(TensorView attention, TensorView gate, TensorView out) {
  CHECK_DEVICE(attention, out);
  CHECK_DEVICE(gate, out);
  CHECK_DIM(3, attention);
  CHECK_DIM(3, gate);
  CHECK_DIM(3, out);
  TVM_FFI_ICHECK_EQ(attention.dtype(), out.dtype()) << "attention and out share one dtype";
  TVM_FFI_ICHECK_EQ(gate.dtype(), out.dtype()) << "the gate is read at the output dtype";

  const int64_t rows = out.size(0);
  const int64_t num_heads = out.size(1);
  const int64_t head_dim = out.size(2);
  // A plan with a fixed row count leaves padding rows behind, so the attention
  // buffer is allowed to be taller than the output. It is never shorter.
  TVM_FFI_ICHECK_GE(attention.size(0), rows) << "attention holds at least the output's rows";
  TVM_FFI_ICHECK_EQ(gate.size(0), rows) << "one gate row per output row";
  TVM_FFI_ICHECK_EQ(attention.size(1), num_heads) << "attention and out agree on heads";
  TVM_FFI_ICHECK_EQ(gate.size(1), num_heads) << "the gate carries one value per head";
  TVM_FFI_ICHECK_EQ(attention.size(2), head_dim) << "attention and out agree on the head";
  TVM_FFI_ICHECK_EQ(gate.size(2), head_dim) << "the gate carries one value per feature";
  // The feature axis is the one the kernel walks with a unit stride. The other
  // two are taken as they come -- the gate arrives as a view of a fused
  // projection, so its rows are not contiguous.
  TVM_FFI_ICHECK_EQ(attention.stride(2), 1) << "attention is contiguous along the head";
  TVM_FFI_ICHECK_EQ(gate.stride(2), 1) << "the gate is contiguous along the head";
  TVM_FFI_ICHECK_EQ(out.stride(2), 1) << "out is contiguous along the head";
  // Everything below is narrowed to uint32 for the kernel.
  constexpr int64_t kUint32Max = 4294967295LL;
  TVM_FFI_ICHECK_LE(rows, kUint32Max) << "rows must fit in 32 bits";
  TVM_FFI_ICHECK_LE(num_heads, kUint32Max) << "num_heads must fit in 32 bits";
  TVM_FFI_ICHECK_LE(head_dim, kUint32Max) << "head_dim must fit in 32 bits";

  // Nothing to do, and nothing to check: the span arithmetic below reads the
  // last element of each axis, which does not exist when an axis is empty.
  if (rows == 0 || num_heads == 0 || head_dim == 0) return;

  // The byte range a tensor's elements occupy, as integers -- comparing
  // pointers from different allocations with < is not something the language
  // defines, and the whole point here is that they may be different
  // allocations.
  const auto span = [](const TensorView& t) {
    int64_t last = 0;
    for (int32_t axis = 0; axis < t.ndim(); ++axis) {
      last += (t.size(axis) - 1) * t.stride(axis);
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(t.data_ptr());
    const uintptr_t bytes = static_cast<uintptr_t>(get_element_size(t));
    return std::pair<uintptr_t, uintptr_t>(base, base + (last + 1) * bytes);
  };
  const auto disjoint = [](const std::pair<uintptr_t, uintptr_t>& a,
                           const std::pair<uintptr_t, uintptr_t>& b) {
    return a.first >= b.second || b.first >= a.second;
  };
  const auto attention_span = span(attention);
  const auto gate_span = span(gate);
  const auto out_span = span(out);

  // The gate is read while `out` is written, so the two may not share a byte.
  TVM_FFI_ICHECK(disjoint(gate_span, out_span))
      << "the gate may not overlap the tensor being written";

  // `out` may be `attention` -- scaling in place is the shape a caller with no
  // row padding wants, and reading an element before writing that same element
  // is safe whatever order the threads run in. Anything between the two is not:
  // with `out` shifted against `attention`, one thread can overwrite a value
  // another has yet to read. So either the two do not touch, or they line up
  // exactly, with `out` covering a prefix of the rows.
  const bool exact_in_place =
      attention.data_ptr() == out.data_ptr() && attention.stride(0) == out.stride(0) &&
      attention.stride(1) == out.stride(1) && attention.stride(2) == out.stride(2);
  TVM_FFI_ICHECK(disjoint(attention_span, out_span) || exact_in_place)
      << "out has to be attention itself or hold none of its bytes; a shifted "
         "view of the same storage would have one thread overwrite what "
         "another has yet to read";

  ffi::CUDADeviceGuard device_guard(out.device().device_id);
  const cudaStream_t stream = get_stream(out.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(out.dtype(), c_type, [&] {
    const cudaError_t status = QSAOutputGate<c_type>(
        static_cast<const c_type*>(attention.data_ptr()),
        static_cast<const c_type*>(gate.data_ptr()), static_cast<c_type*>(out.data_ptr()),
        static_cast<uint32_t>(rows), static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(head_dim), attention.stride(0), attention.stride(1), gate.stride(0),
        gate.stride(1), out.stride(0), out.stride(1), stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "qsa_output_gate failed with error " << cudaGetErrorString(status);
    return true;
  });
}

// What this build of the gate carries, as a bitmask the binary answers for
// itself. A cached artifact older than the source would report what it was
// compiled with rather than what the source now says, which is the point: a
// caller asking "can you gate bfloat16" gets the module's answer, not the
// repository's.
//
//   1  float16 output
//   2  bfloat16 output
int64_t qsa_output_gate_capabilities() {
  int64_t bits = 0;
#ifdef FLASHINFER_ENABLE_F16
  bits |= 1;
#endif
#ifdef FLASHINFER_ENABLE_BF16
  bits |= 2;
#endif
  return bits;
}
