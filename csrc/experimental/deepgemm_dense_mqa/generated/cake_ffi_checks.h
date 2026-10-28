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
// Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
// DeepGEMM portions are licensed under MIT; see DEEPGEMM_NOTICE.txt in this directory.

// Host surface of the generated Cake launchers (``*_binding.cu``) for a package built without
// FlashInfer's csrc: the TVM-FFI / DLPack headers the launchers use, the named tensor-argument
// checks and the CUDA runtime version the library was compiled against.
#pragma once

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>
#include <cuda_runtime_api.h>

inline void check_cuda_tensor(const tvm::ffi::TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor, got device_type=" << static_cast<int>(t.device().device_type);
}

inline void check_dtype(const tvm::ffi::TensorView& t, DLDataType expected, const char* name) {
  DLDataType d = t.dtype();
  TVM_FFI_CHECK(d.code == expected.code && d.bits == expected.bits && d.lanes == expected.lanes, TypeError)
      << name << " dtype mismatch: expected DLDataType(code=" << static_cast<int>(expected.code)
      << ", bits=" << static_cast<int>(expected.bits) << ", lanes=" << static_cast<int>(expected.lanes)
      << "), got (code=" << static_cast<int>(d.code) << ", bits=" << static_cast<int>(d.bits)
      << ", lanes=" << static_cast<int>(d.lanes) << ")";
}

inline void check_contiguous(const tvm::ffi::TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.IsContiguous(), ValueError) << name << " must be contiguous";
}

inline void check_same_device(const tvm::ffi::TensorView& t, const tvm::ffi::TensorView& reference,
                              const char* name, const char* reference_name) {
  TVM_FFI_CHECK(t.device().device_type == reference.device().device_type &&
                    t.device().device_id == reference.device().device_id,
                ValueError)
      << name << " must be on the same device as " << reference_name << ": got device_type="
      << static_cast<int>(t.device().device_type) << " id=" << t.device().device_id
      << " versus device_type=" << static_cast<int>(reference.device().device_type)
      << " id=" << reference.device().device_id;
}

// Require the dimensions above the last `trailing` dimensions to form one dense
// row-major chain, so they can be folded into a single logical dimension whose
// step is stride(-(trailing + 1)). Shape products are stride-independent, so
// this verifies the physical layout instead of inventing a folded stride.
inline void check_dense_leading_dims(const tvm::ffi::TensorView& t, int trailing, const char* name) {
  TVM_FFI_CHECK(trailing > 0 && t.ndim() >= trailing, ValueError)
      << name << " cannot fold leading dimensions above " << trailing << " trailing dims from ndim=" << t.ndim();
  int outer_last = t.ndim() - trailing - 1;
  if (outer_last <= 0) {
    return;
  }
  int64_t step = t.stride(outer_last);
  TVM_FFI_CHECK(step > 0, ValueError) << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      TVM_FFI_CHECK(t.stride(axis) == expected, ValueError)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis) << ", expected " << expected;
    }
  }
}

// The CUDA runtime this library was compiled against (``CUDART_VERSION``, e.g. 13000 for 13.0); every
// binding exports it as ``build_cuda_version`` so the package loader can compare it with the prebuilt
// sidecar before the first launch.
inline int64_t cake_ffi_build_cuda_version() { return CUDART_VERSION; }
