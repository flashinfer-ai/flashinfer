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
// Issue the launches of a multi-kernel route from one FFI call.
//
// run_sequence(fn_0, n_0, arg_0_0, ..., arg_0_{n_0-1}, fn_1, n_1, ...) calls
// every generated launcher fn_i with its n_i arguments, in order, inside a
// single Python-to-C transition. The arguments are converted once (the
// current stream is recorded from the first device tensor as for any FFI
// call) and nothing runs on the host between two consecutive launches, so a
// short producer kernel is never left waiting for its reducer's host work.
#include <tvm/ffi/any.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

#include <cstdint>

extern "C" {
TVM_FFI_DLL_EXPORT int __tvm_ffi_run_sequence(void* self, const TVMFFIAny* args, int32_t num_args,
                                              TVMFFIAny* result) {
  TVM_FFI_SAFE_CALL_BEGIN();
  const tvm::ffi::AnyView* view = reinterpret_cast<const tvm::ffi::AnyView*>(args);
  int32_t i = 0;
  while (i < num_args) {
    TVM_FFI_CHECK(i + 2 <= num_args, ValueError)
        << "run_sequence: launcher " << i << " lacks its argument count";
    tvm::ffi::Function launcher = view[i].cast<tvm::ffi::Function>();
    int64_t count = view[i + 1].cast<int64_t>();
    TVM_FFI_CHECK(count >= 0 && i + 2 + count <= num_args, ValueError)
        << "run_sequence: launcher at " << i << " declares " << count << " arguments, "
        << (num_args - i - 2) << " remain";
    tvm::ffi::Any ignored;
    launcher.CallPacked(view + i + 2, static_cast<int32_t>(count), &ignored);
    i += 2 + static_cast<int32_t>(count);
  }
  TVM_FFI_SAFE_CALL_END();
}
}
