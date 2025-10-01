/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include "flashinfer/logging.h"

#include <tvm/ffi/function.h>

#include "Python.h"

void set_log_level(int64_t log_level_code) {
  auto log_level = static_cast<spdlog::level::level_enum>(log_level_code);
  flashinfer::logging::set_log_level(log_level);
}

void try_log_info(const std::string& msg) { FLASHINFER_LOG_INFO(msg); }

TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_log_level, set_log_level);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(try_log_info, try_log_info);
