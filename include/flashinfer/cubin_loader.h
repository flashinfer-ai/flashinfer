/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Cubin loading via TVM-FFI.
//
// The Python side registers a TVM-FFI global function "flashinfer.get_cubin"
// that downloads/caches cubin files and returns their bytes.
// C++ code calls getCubin() which invokes this function through TVM-FFI's
// cross-language function call mechanism — no ctypes callbacks, no global
// function pointers, no thread-local storage needed.

#pragma once

#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <stdexcept>
#include <string>

// Get cubin bytes from the Python-registered TVM-FFI function.
// The Python function handles downloading, caching, and SHA256 verification.
inline std::string getCubin(const std::string& name, const std::string& sha256) {
  static tvm::ffi::Function func = tvm::ffi::Function::GetGlobalRequired("flashinfer.get_cubin");
  tvm::ffi::Bytes result = func(name, sha256).cast<tvm::ffi::Bytes>();
  return std::string(result.data(), result.size());
}
