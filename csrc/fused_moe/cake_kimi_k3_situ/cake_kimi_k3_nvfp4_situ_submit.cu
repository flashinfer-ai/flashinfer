/*
 * Copyright (c) 2023 by FlashInfer team.
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

// Route submitter of the Cake SiTU backend: one host-only tvm-ffi entry that
// drives the prepared stage shims of a route in two phases. The plan is a flat
// argument list of `[prepare, submit, argc, args...]` per stage, where
// `prepare` and `submit` are the addresses returned by the stage program's
// `prepare_entry` / `submit_entry` exports (int64). Every stage of the call is
// prepared (arguments validated, tensor maps encoded, kernel parameters
// packed) before the first stage is submitted, so the launches reach the
// stream back to back as they do from a prepared sequence binding. The entry
// holds no state and launches nothing itself.
#include <cstdint>

#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

namespace {

using PrepareStageFn = int (*)(const TVMFFIAny*, int32_t, void*);
using SubmitStageFn = int (*)(void*);

constexpr int32_t kMaxStages = 16;

struct Stage {
  PrepareStageFn prepare;
  SubmitStageFn submit;
  const TVMFFIAny* args;
  int32_t argc;
};

// The device of the first tensor among a stage's arguments, or -1 when the
// stage binds no tensor.
int FirstTensorDevice(const tvm::ffi::AnyView* args, int32_t argc, int* device_type) {
  for (int32_t i = 0; i < argc; ++i) {
    // try_cast: tensors arrive as owned Tensor objects or DLTensor pointers, both viewable.
    if (auto tensor = args[i].try_cast<tvm::ffi::TensorView>()) {
      *device_type = tensor->device().device_type;
      return tensor->device().device_id;
    }
  }
  return -1;
}

}  // namespace

extern "C" {
TVM_FFI_DLL_EXPORT int __tvm_ffi_run(void* self, const TVMFFIAny* raw_args, int32_t num_args,
                                     TVMFFIAny* raw_result) {
  TVM_FFI_SAFE_CALL_BEGIN();
  (void)self;
  const tvm::ffi::AnyView* args = reinterpret_cast<const tvm::ffi::AnyView*>(raw_args);
  tvm::ffi::Any* result = reinterpret_cast<tvm::ffi::Any*>(raw_result);
  Stage stages[kMaxStages];
  int32_t num_stages = 0;
  int32_t index = 0;
  while (index < num_args) {
    TVM_FFI_CHECK(index + 2 < num_args, ValueError)
        << "stage plan truncated at entry " << index << " of " << num_args;
    TVM_FFI_CHECK(num_stages < kMaxStages, ValueError)
        << "stage plan declares more than " << kMaxStages << " stages";
    int64_t prepare = args[index].cast<int64_t>();
    int64_t submit = args[index + 1].cast<int64_t>();
    int64_t argc = args[index + 2].cast<int64_t>();
    TVM_FFI_CHECK(prepare > 0 && submit > 0, ValueError)
        << "stage plan entry " << index << " has no prepare/submit entry points";
    TVM_FFI_CHECK(argc >= 0 && index + 3 + argc <= num_args, ValueError)
        << "stage plan declares " << argc << " arguments at entry " << index << " but " << num_args
        << " values were passed";
    for (int32_t previous = 0; previous < num_stages; ++previous) {
      TVM_FFI_CHECK(reinterpret_cast<int64_t>(stages[previous].prepare) != prepare, ValueError)
          << "stage plan drives one stage program twice; its prepared state is per program";
    }
    stages[num_stages++] = Stage{reinterpret_cast<PrepareStageFn>(static_cast<uintptr_t>(prepare)),
                                 reinterpret_cast<SubmitStageFn>(static_cast<uintptr_t>(submit)),
                                 raw_args + index + 3, static_cast<int32_t>(argc)};
    index += 3 + static_cast<int32_t>(argc);
  }
  TVM_FFI_CHECK(num_stages > 0, ValueError) << "stage plan is empty";

  // One device context and stream for the whole call; every stage's first
  // tensor must live on it.
  int device_type = kDLCUDA;
  int device_id = FirstTensorDevice(args + 3, stages[0].argc, &device_type);
  TVM_FFI_CHECK(device_id >= 0, ValueError) << "the first stage binds no tensor to select the device";
  for (int32_t s = 1; s < num_stages; ++s) {
    int other_type = kDLCUDA;
    int other = FirstTensorDevice(reinterpret_cast<const tvm::ffi::AnyView*>(stages[s].args), stages[s].argc,
                                  &other_type);
    TVM_FFI_CHECK(other < 0 || (other == device_id && other_type == device_type), ValueError)
        << "stage " << s << " binds tensors on device " << other << " but the call runs on device "
        << device_id;
  }
  tvm::ffi::CUDADeviceGuard device_guard(device_id);
  {
    // Bind the primary context for the tensor-map encoders once per thread and device.
    static thread_local int encoder_context_device = -1;
    if (encoder_context_device != device_id) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(device_id));
      encoder_context_device = device_id;
    }
  }
  void* stream = TVMFFIEnvGetStream(device_type, device_id);

  for (int32_t s = 0; s < num_stages; ++s) {
    int status = stages[s].prepare(stages[s].args, stages[s].argc, stream);
    if (status != 0) return status;
  }
  for (int32_t s = 0; s < num_stages; ++s) {
    int status = stages[s].submit(stream);
    if (status != 0) return status;
  }
  *result = tvm::ffi::Any();
  TVM_FFI_SAFE_CALL_END();
}
}
