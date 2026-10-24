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

// Carrier of the device cubins of one Cake SM120 MegaMoE parameter set and transport
// flavor (hand-written; not part of the generated closure).  The JIT embeds the M128 and
// M64 instances of the flavor (embedded_cubin_factory), so a device build is an ordinary
// JIT module that flashinfer/aot.py can package.  The images are handed back to the host
// binding (module_load_function), which loads them with the driver API for the
// cooperative launch.

#include <tvm/ffi/extra/cuda/cubin_launcher.h>

#include <cstddef>
#include <cstdint>

#include "tvm_ffi_utils.h"

TVM_FFI_EMBED_CUBIN(cake_sm120_megamoe_m128);
TVM_FFI_EMBED_CUBIN(cake_sm120_megamoe_m64);

namespace flashinfer::cake_sm120_megamoe {

// The cubin image of the row schedule with tile height ``task_m`` (128 or 64).
tvm::ffi::Bytes cubin_image(int64_t task_m) {
  const char* begin = nullptr;
  const char* end = nullptr;
  if (task_m == 128) {
    begin = __tvm_ffi__cubin_cake_sm120_megamoe_m128;
    end = __tvm_ffi__cubin_cake_sm120_megamoe_m128_end;
  } else if (task_m == 64) {
    begin = __tvm_ffi__cubin_cake_sm120_megamoe_m64;
    end = __tvm_ffi__cubin_cake_sm120_megamoe_m64_end;
  } else {
    TVM_FFI_THROW(ValueError) << "task_m must be 128 or 64, got " << task_m;
  }
  return tvm::ffi::Bytes(begin, static_cast<size_t>(end - begin));
}

}  // namespace flashinfer::cake_sm120_megamoe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cubin_image, flashinfer::cake_sm120_megamoe::cubin_image);
