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
// clang-format off
// Tensor checks shared by the Cake SM120 sparse-MLA host bindings
// (cake_sparse_mla_dsv4_nvfp4_binding.cu and cake_sparse_mla_dsv41_mixed_binding.cu).
//
// Verbatim fragment, not a free-standing header: each binding includes it from inside its anonymous
// namespace, after "tvm_ffi_utils.h" (TensorView, DLDevice, DLDataType, the dl_* dtype constants,
// encode_dlpack_dtype, TVM_FFI_CHECK) and after the family's *_kernels.h, at the exact position where
// these definitions used to stand.  It therefore opens no namespace and includes nothing itself; the
// preprocessed text of each binding is unchanged apart from line markers.  The messages are those of
// the sparse-MLA bindings (the reference tensor is q); the RoPE-insert binding keeps its own
// CheckDevice (reference tensor kv).  CheckCache (page payload per family) and LengthPointer stay in
// the bindings.
#pragma once

void CheckDevice(const TensorView& tensor, const char* name, DLDevice device, int64_t alignment) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError) << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as q";
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0, ValueError)
      << name << " must be " << alignment << "-byte aligned";
}

void CheckDtype(const TensorView& tensor, const char* name, DLDataType dtype, const char* dtype_name) {
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype), ValueError)
      << name << " must be " << dtype_name;
}

void CheckRows(const TensorView& tensor, const char* name, int64_t tokens, int64_t heads, int64_t dim, DLDevice device) {
  CheckDevice(tensor, name, device, 16);
  CheckDtype(tensor, name, dl_bfloat16, "bfloat16");
  TVM_FFI_CHECK(tensor.ndim() == 3 && tensor.size(0) == tokens && tensor.size(1) == heads && tensor.size(2) == dim,
                ValueError)
      << name << " must have shape [" << tokens << ", " << heads << ", " << dim << "]";
  TVM_FFI_CHECK(tensor.IsContiguous(), ValueError) << name << " must be contiguous";
}

void CheckLse(const TensorView& tensor, const char* name, int64_t tokens, int64_t heads, DLDevice device) {
  CheckDevice(tensor, name, device, 4);
  CheckDtype(tensor, name, dl_float32, "float32");
  TVM_FFI_CHECK(tensor.ndim() == 2 && tensor.size(0) == tokens && tensor.size(1) == heads, ValueError)
      << name << " must have shape [" << tokens << ", " << heads << "]";
  TVM_FFI_CHECK(tensor.IsContiguous(), ValueError) << name << " must be contiguous";
}

void CheckIndices(const TensorView& tensor, const char* name, int64_t tokens, DLDevice device) {
  CheckDevice(tensor, name, device, 4);
  CheckDtype(tensor, name, dl_int32, "int32");
  TVM_FFI_CHECK(tensor.ndim() == 2 && tensor.size(0) == tokens && tensor.size(1) >= 1, ValueError)
      << name << " must have shape [" << tokens << ", topk] with topk >= 1";
  TVM_FFI_CHECK(tensor.IsContiguous(), ValueError) << name << " must be contiguous";
}
// clang-format on
