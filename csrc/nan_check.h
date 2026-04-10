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
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flashinfer {
namespace nan_check {

bool IsEnabled();
bool ShouldTrap();

void LaunchNanCheckFloat(const void* data, int64_t numel, const char* label, cudaStream_t stream);
void LaunchNanCheckHalf(const void* data, int64_t numel, const char* label, cudaStream_t stream);
void LaunchNanCheckBFloat16(const void* data, int64_t numel, const char* label,
                            cudaStream_t stream);
void LaunchNanCheckFp8Bytes(const void* data, int64_t num_bytes, const char* label,
                            cudaStream_t stream);

}  // namespace nan_check
}  // namespace flashinfer
