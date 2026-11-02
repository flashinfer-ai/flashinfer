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

// Host helpers shared by the generated tvm-ffi bindings of this package's TMA tree kernels.
// The fragment is included once per binding inside its `cake_binding_<kernel>` namespace after
// KERNEL_NAME, so it has no include guard by design; hand-folded pending exporter regeneration.

void CheckCuda(cudaError_t status, const char* what) {
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << KERNEL_NAME << ": " << what << " failed: " << cudaGetErrorString(status);
}

void CheckCu(CUresult result, const char* what) {
  if (result != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    TVM_FFI_THROW(RuntimeError) << KERNEL_NAME << ": " << what << " failed with "
                                << (name ? name : "an unknown CUresult") << " (" << static_cast<int>(result)
                                << "): " << (message ? message : "");
  }
}

void* DataPtr(const TensorView& t) { return static_cast<char*>(t.data_ptr()) + t.byte_offset(); }

void CheckUnitInnerStride(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << KERNEL_NAME << ": '" << name << "' needs a unit innermost stride, got " << t.stride(-1);
}

// The product of the extents above the trailing `trailing` dims.
int64_t NumelAbove(const TensorView& t, int trailing) {
  int64_t product = 1;
  for (int axis = 0; axis < t.ndim() - trailing; ++axis) product *= t.size(axis);
  return product;
}

cuuint64_t Extent(int64_t value, const char* descriptor) {
  TVM_FFI_CHECK(value > 0, ValueError)
      << KERNEL_NAME << ": an extent of descriptor '" << descriptor << "' resolves to " << value;
  return static_cast<cuuint64_t>(value);
}

cuuint64_t Stride(int64_t value, const char* descriptor) {
  TVM_FFI_CHECK(value >= 0, ValueError)
      << KERNEL_NAME << ": a stride of descriptor '" << descriptor << "' resolves to " << value;
  return static_cast<cuuint64_t>(value);
}

template <int N, int A>
struct alignas(A) CakeRecord {
  unsigned char bytes[N];
};

template <int N, int A, typename T>
void PutField(CakeRecord<N, A>& record, int offset, T value) {
  std::memcpy(record.bytes + offset, &value, sizeof(T));
}
