/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <filesystem>
#include <stdexcept>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace deep_gemm::jit {

class CacheFileLock {
 public:
  explicit CacheFileLock(std::filesystem::path const& path) {
#ifndef _WIN32
    fd_ = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
    if (fd_ < 0) {
      throw std::runtime_error("Failed to lock DeepGEMM JIT cache");
    }
    if (flock(fd_, LOCK_EX) != 0) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("Failed to lock DeepGEMM JIT cache");
    }
#endif
  }

  ~CacheFileLock() {
#ifndef _WIN32
    if (fd_ >= 0) {
      flock(fd_, LOCK_UN);
      close(fd_);
    }
#endif
  }

  CacheFileLock(CacheFileLock const&) = delete;
  CacheFileLock& operator=(CacheFileLock const&) = delete;

 private:
#ifndef _WIN32
  int fd_{-1};
#endif
};

inline bool isRegularNonEmptyFile(std::filesystem::path const& path) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    return false;
  }
  return std::filesystem::file_size(path, error) > 0 && !error;
}

}  // namespace deep_gemm::jit
