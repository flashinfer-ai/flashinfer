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

// The device builds of the Blackwell softmax kernels that read the PDL flag
// (bit 2 of parameter_kind) and issue griddepcontrol.wait /
// launch_dependents: sm_103a and sm_107a, the architectures the dispatcher
// launches with PDL. Other builds never receive the flag.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1030 || __CUDA_ARCH__ == 1070)
#define FLASHINFER_CAKE_SOFTMAX_DEVICE_PDL 1
#else
#define FLASHINFER_CAKE_SOFTMAX_DEVICE_PDL 0
#endif

// A global load the compiler keeps after a preceding griddepcontrol.wait. The
// per-row temperature may be written by the prerequisite grid, and a plain
// load of a __restrict__ pointer that the kernel never writes becomes
// ld.global.nc, which the compiler may hoist above the wait.
__device__ __forceinline__ float cake_softmax_ld_ordered(const float* address) {
  float value;
  asm volatile("ld.global.f32 %0, [%1];" : "=f"(value) : "l"(address) : "memory");
  return value;
}
