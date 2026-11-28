/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <tvm/ffi/container/array.h>

#include "common.h"
#include "tvm_ffi_utils.h"

namespace flashinfer::glm5_tp8::binding {
using tvm::ffi::Array;
using tvm::ffi::Tensor;

inline void check_tensor(const Tensor& value, DLDataType dtype, DLDevice device, int64_t elements,
                         const char* name, uintptr_t alignment = 16) {
  CHECK_CUDA(value);
  CHECK_CONTIGUOUS(value);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(value.dtype()), encode_dlpack_dtype(dtype))
      << name << " has the wrong dtype";
  TVM_FFI_ICHECK_EQ(value.device().device_id, device.device_id)
      << name << " is on the wrong device";
  TVM_FFI_ICHECK_GE(value.numel(), elements) << name << " is too small";
  if (elements)
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(value.data_ptr()) % alignment, 0)
        << name << " has invalid alignment";
}

template <class T>
T* pointer(const Tensor& value) {
  return static_cast<T*>(value.data_ptr());
}

inline Array<int64_t> resources(KernelResources r) {
  return {r.grid,        r.block,        r.dynamic_shared_bytes, r.registers, r.static_shared_bytes,
          r.local_bytes, r.blocks_per_sm};
}

inline void check_peers(const Array<int64_t>& peers) {
  TVM_FFI_ICHECK_EQ(peers.size(), 8) << "expected eight peer addresses";
  for (auto p : peers) TVM_FFI_ICHECK(p > 0 && p % 16 == 0) << "invalid peer address/alignment";
}
}  // namespace flashinfer::glm5_tp8::binding
