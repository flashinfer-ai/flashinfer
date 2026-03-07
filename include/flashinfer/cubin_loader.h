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

// Cubin loading via TVM-FFI CubinModule.
//
// The Python side registers a TVM-FFI global function "flashinfer.get_cubin"
// that downloads/caches cubin files and returns their bytes.
//
// C++ code uses getCubinModule() to get a CubinModule (RAII-managed via
// tvm::ffi::CubinModule) and getCubinKernel() to get a CubinKernel handle.
// For kernels that need cuLaunchKernelEx (cluster dimensions, PDL), extract
// the raw handle via CubinKernel::GetHandle() and cast to CUfunction.

#pragma once

#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace flashinfer {
namespace cubin_loader {

// Get cubin bytes from the Python-registered TVM-FFI function.
inline std::string getCubinBytes(const std::string& name, const std::string& sha256) {
  static tvm::ffi::Function func = tvm::ffi::Function::GetGlobalRequired("flashinfer.get_cubin");
  tvm::ffi::Bytes result = func(name, sha256).cast<tvm::ffi::Bytes>();
  return std::string(result.data(), result.size());
}

// Thread-safe cache of CubinModules keyed by cubin path.
// CubinModule is move-only, so we store shared_ptr for safe sharing.
class CubinModuleCache {
 public:
  static CubinModuleCache& Instance() {
    static CubinModuleCache instance;
    return instance;
  }

  // Get or load a CubinModule for the given cubin path + sha256.
  std::shared_ptr<tvm::ffi::CubinModule> GetModule(const std::string& path,
                                                   const std::string& sha256) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = modules_.find(path);
    if (it != modules_.end()) {
      return it->second;
    }
    std::string cubin = getCubinBytes(path, sha256);
    if (cubin.empty()) {
      throw std::runtime_error("Failed to load cubin: " + path);
    }
    auto mod = std::make_shared<tvm::ffi::CubinModule>(tvm::ffi::Bytes(cubin.data(), cubin.size()));
    modules_[path] = mod;
    return mod;
  }

 private:
  CubinModuleCache() = default;
  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<tvm::ffi::CubinModule>> modules_;
};

// Get a CubinModule for a cubin file (cached, thread-safe).
inline std::shared_ptr<tvm::ffi::CubinModule> getCubinModule(const std::string& path,
                                                             const std::string& sha256) {
  return CubinModuleCache::Instance().GetModule(path, sha256);
}

// Get a CubinKernel from a cubin file by kernel name.
// For kernels needing >48KB shared memory, use smem_bytes to set the max.
inline tvm::ffi::CubinKernel getCubinKernel(const std::string& cubin_path,
                                            const std::string& sha256,
                                            const std::string& kernel_name,
                                            int64_t smem_bytes = 0) {
  auto mod = getCubinModule(cubin_path, sha256);
  if (smem_bytes >= 48 * 1024) {
    return mod->GetKernelWithMaxDynamicSharedMemory(kernel_name.c_str(), smem_bytes);
  }
  return mod->GetKernel(kernel_name.c_str());
}

}  // namespace cubin_loader

// Legacy API: get raw cubin bytes (for code that still uses cuModuleLoadData directly).
inline std::string getCubin(const std::string& name, const std::string& sha256) {
  return cubin_loader::getCubinBytes(name, sha256);
}

}  // namespace flashinfer
