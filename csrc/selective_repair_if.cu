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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <flashinfer/attention/selective_repair_if.cuh>
#include <memory>

#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace {

void CheckCuda(cudaError_t status, const char* step = "selective repair CUDA call") {
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << step << ": " << cudaGetErrorString(status);
}

// PyTorch's round-robin stream pool can alias the active parent capture stream;
// beginning the conditional body capture on that handle fails with CUDA 401.
// Own a non-pooled stream from planning through graph use. Python's ExternalStream
// borrows this handle; it must not create or destroy the body stream on replay.
struct RepairBodyStream {
  const int device;
  cudaStream_t stream = nullptr;

  explicit RepairBodyStream(int device) : device(device) {
    ffi::CUDADeviceGuard guard(device);
    cudaFuncAttributes attributes;
    CheckCuda(
        cudaFuncGetAttributes(&attributes, flashinfer::attention::SetSelectiveRepairCondition),
        "materialize selective repair predicate");
    CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  ~RepairBodyStream() noexcept {
    // Destruction is asynchronous and must not throw during interpreter teardown.
    int previous;
    if (cudaGetDevice(&previous) != cudaSuccess) return;
    if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
    cudaStreamCaptureStatus status;
    if (cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
        status != cudaStreamCaptureStatusNone) {
      cudaGraph_t graph = nullptr;
      cudaStreamEndCapture(stream, &graph);
    }
    cudaStreamDestroy(stream);
    if (previous != device) cudaSetDevice(previous);
  }
};

tvm::ffi::Function CreateSelectiveRepairBodyStream(int64_t device) {
  auto owner = std::make_shared<RepairBodyStream>(static_cast<int>(device));
  return tvm::ffi::Function::FromTyped(
      [owner]() -> int64_t { return reinterpret_cast<int64_t>(owner->stream); });
}

int64_t SelectiveRepairBodyCaptureStatus(TensorView anchor, int64_t stream_handle) {
  ffi::CUDADeviceGuard guard(anchor.device().device_id);
  cudaStreamCaptureStatus status;
  CheckCuda(cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(stream_handle), &status));
  return static_cast<int64_t>(status);
}

void PreflightSelectiveRepairBody(TensorView anchor, int64_t stream_handle) {
  ffi::CUDADeviceGuard guard(anchor.device().device_id);
  TVM_FFI_ICHECK(reinterpret_cast<cudaStream_t>(stream_handle) != get_stream(anchor.device()))
      << "repair body stream must differ from the parent capture stream";
  TVM_FFI_ICHECK_EQ(SelectiveRepairBodyCaptureStatus(anchor, stream_handle),
                    static_cast<int64_t>(cudaStreamCaptureStatusNone))
      << "repair body stream must be idle before parent graph mutation";
}

void AbortSelectiveRepairBody(TensorView anchor, int64_t stream_handle) {
  ffi::CUDADeviceGuard guard(anchor.device().device_id);
  auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
  cudaStreamCaptureStatus status;
  CheckCuda(cudaStreamIsCapturing(stream, &status));
  if (status == cudaStreamCaptureStatusNone) return;
  cudaGraph_t graph = nullptr;
  auto error = cudaStreamEndCapture(stream, &graph);
  // BeginCaptureToGraph borrows the conditional's graph; never destroy it here.
  if (error != cudaErrorStreamCaptureInvalidated) CheckCuda(error);
}

void CaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* status, unsigned long long* id,
                 cudaGraph_t* graph, const cudaGraphNode_t** dependencies, size_t* count,
                 const char* step) {
#if CUDART_VERSION >= 13000
  CheckCuda(cudaStreamGetCaptureInfo(stream, status, id, graph, dependencies, nullptr, count),
            step);
#else
  CheckCuda(cudaStreamGetCaptureInfo_v3(stream, status, id, graph, dependencies, nullptr, count),
            step);
#endif
  TVM_FFI_ICHECK_NE(*status, cudaStreamCaptureStatusInvalidated)
      << "selective repair capture invalidated after " << step;
}

// These host queries neither synchronize nor add graph nodes. Check each mutation
// before the next API can obscure its failure with a downstream capture error.
void CheckCapture(cudaStream_t stream, const char* step) {
  cudaStreamCaptureStatus status;
  unsigned long long id;
  cudaGraph_t graph;
  const cudaGraphNode_t* dependencies;
  size_t count;
  CaptureInfo(stream, &status, &id, &graph, &dependencies, &count, step);
}

void CheckVector(TensorView tensor, int device, int64_t size) {
  CHECK_INPUT(tensor);
  CHECK_INPUT_TYPE(tensor, dl_int32);
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 1);
  TVM_FFI_ICHECK_EQ(tensor.size(0), size);
  TVM_FFI_ICHECK_EQ(tensor.device().device_id, device);
}

int32_t* CountPointer(Optional<TensorView> count, int device) {
  if (!count.has_value()) return nullptr;
  CheckVector(count.value(), device, 1);
  return static_cast<int32_t*>(count.value().data_ptr());
}

}  // namespace

Array<int64_t> BeginSelectiveRepairIf(TensorView flags, TensorView second_flags,
                                      TensorView exact_error, Optional<TensorView> first_count,
                                      Optional<TensorView> second_count) {
  const int device = flags.device().device_id;
  CheckVector(flags, device, flags.numel());
  CheckVector(second_flags, device, flags.numel());
  CheckVector(exact_error, device, 1);
  auto* first_ptr = CountPointer(first_count, device);
  auto* second_ptr = CountPointer(second_count, device);
  ffi::CUDADeviceGuard guard(device);
  const auto stream = get_stream(flags.device());
  cudaStreamCaptureStatus status;
  unsigned long long id;
  cudaGraph_t graph;
  const cudaGraphNode_t* dependencies;
  size_t count;
  CaptureInfo(stream, &status, &id, &graph, &dependencies, &count, "begin_if entry");
  TVM_FFI_ICHECK_EQ(status, cudaStreamCaptureStatusActive);
  cudaGraphConditionalHandle handle;
  CheckCuda(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault),
            "cudaGraphConditionalHandleCreate");
  CheckCapture(stream, "cudaGraphConditionalHandleCreate");
  flashinfer::attention::SetSelectiveRepairCondition<<<1, 256, 0, stream>>>(
      static_cast<const int32_t*>(flags.data_ptr()), static_cast<int32_t*>(second_flags.data_ptr()),
      static_cast<int32_t*>(exact_error.data_ptr()), first_ptr, second_ptr, flags.numel(), handle);
  CheckCuda(cudaGetLastError(), "SetSelectiveRepairCondition launch");
  CaptureInfo(stream, &status, &id, &graph, &dependencies, &count,
              "SetSelectiveRepairCondition launch");
  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeIf;
  params.conditional.size = 1;
  cudaGraphNode_t node;
#if CUDART_VERSION >= 13000
  CheckCuda(cudaGraphAddNode(&node, graph, dependencies, nullptr, count, &params),
            "cudaGraphAddNode conditional IF");
#else
  CheckCuda(cudaGraphAddNode_v2(&node, graph, dependencies, nullptr, count, &params),
            "cudaGraphAddNode conditional IF");
#endif
  CheckCapture(stream, "cudaGraphAddNode conditional IF");
  // CUDA owns the body graph and handle for the lifetime of this parent graph.
  return {static_cast<int64_t>(id), reinterpret_cast<int64_t>(node),
          reinterpret_cast<int64_t>(params.conditional.phGraph_out[0])};
}

void BeginSelectiveRepairBody(TensorView anchor, int64_t stream_handle, int64_t graph_handle) {
  ffi::CUDADeviceGuard guard(anchor.device().device_id);
  CheckCuda(cudaStreamBeginCaptureToGraph(reinterpret_cast<cudaStream_t>(stream_handle),
                                          reinterpret_cast<cudaGraph_t>(graph_handle), nullptr,
                                          nullptr, 0, cudaStreamCaptureModeRelaxed),
            "cudaStreamBeginCaptureToGraph repair body");
  CheckCapture(reinterpret_cast<cudaStream_t>(stream_handle),
               "cudaStreamBeginCaptureToGraph repair body");
  CheckCapture(get_stream(anchor.device()), "repair body begin parent");
}

void EndSelectiveRepairBody(TensorView anchor, int64_t stream_handle, int64_t graph_handle) {
  ffi::CUDADeviceGuard guard(anchor.device().device_id);
  cudaGraph_t graph;
  CheckCapture(reinterpret_cast<cudaStream_t>(stream_handle), "repair body operations");
  CheckCuda(cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(stream_handle), &graph),
            "cudaStreamEndCapture repair body");
  CheckCapture(reinterpret_cast<cudaStream_t>(stream_handle), "cudaStreamEndCapture repair body");
  CheckCapture(get_stream(anchor.device()), "repair body end parent");
  TVM_FFI_ICHECK(graph == reinterpret_cast<cudaGraph_t>(graph_handle));
}

void FinishSelectiveRepairIf(TensorView anchor, Array<int64_t> state) {
  TVM_FFI_ICHECK_EQ(state.size(), 3);
  ffi::CUDADeviceGuard guard(anchor.device().device_id);
  const auto stream = get_stream(anchor.device());
  cudaStreamCaptureStatus status;
  unsigned long long id;
  cudaGraph_t graph;
  const cudaGraphNode_t* dependencies;
  size_t count;
  CaptureInfo(stream, &status, &id, &graph, &dependencies, &count, "finish_if entry");
  TVM_FFI_ICHECK(status == cudaStreamCaptureStatusActive && id == state[0]);
  auto node = reinterpret_cast<cudaGraphNode_t>(state[1]);
  // Work captured after run() must depend on the whole conditional, including
  // an executed repair body, rather than just the predicate kernel.
#if CUDART_VERSION >= 13000
  CheckCuda(cudaStreamUpdateCaptureDependencies(stream, &node, nullptr, 1,
                                                cudaStreamSetCaptureDependencies),
            "cudaStreamUpdateCaptureDependencies");
#else
  CheckCuda(cudaStreamUpdateCaptureDependencies_v2(stream, &node, nullptr, 1,
                                                   cudaStreamSetCaptureDependencies),
            "cudaStreamUpdateCaptureDependencies");
#endif
  CheckCapture(stream, "cudaStreamUpdateCaptureDependencies");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(begin_if, BeginSelectiveRepairIf);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(create_body_stream, CreateSelectiveRepairBodyStream);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(preflight_body, PreflightSelectiveRepairBody);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(body_capture_status, SelectiveRepairBodyCaptureStatus);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(abort_body, AbortSelectiveRepairBody);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(begin_body, BeginSelectiveRepairBody);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(end_body, EndSelectiveRepairBody);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(finish_if, FinishSelectiveRepairIf);
