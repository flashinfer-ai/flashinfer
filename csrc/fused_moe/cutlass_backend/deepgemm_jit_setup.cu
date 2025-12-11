/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <tvm/ffi/extra/module.h>

#include <filesystem>

#include "nv_internal/tensorrt_llm/deep_gemm/compiler.cuh"

namespace flashinfer {

void set_deepgemm_jit_include_dirs(tvm::ffi::Array<tvm::ffi::String> include_dirs) {
  std::vector<std::filesystem::path> dirs;
  for (const auto& dir : include_dirs) {
    dirs.push_back(std::filesystem::path(std::string(dir)));
  }
  deep_gemm::jit::Compiler::setIncludeDirs(dirs);
}

}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_deepgemm_jit_include_dirs,
                              flashinfer::set_deepgemm_jit_include_dirs);
