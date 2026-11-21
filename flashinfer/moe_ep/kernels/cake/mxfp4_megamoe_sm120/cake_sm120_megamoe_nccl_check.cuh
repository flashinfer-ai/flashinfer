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

// Hand-written; not part of the generated closure. shim/cake_jit.py force-includes it (nvcc
// -include) ahead of every device instance, so it sees the same NCCL headers and -D switches as the
// generated source.
#pragma once

// Device passes only: the NCCL Device API is device code (the nosync MegaMoE includes it in its
// device pass only, and with every GIN backend switched off a host pass cannot compile it).
#if defined(__CUDA_ARCH__)

#ifndef _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE
#define _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE
#endif
#include <nccl_device.h>

static_assert(NCCL_VERSION_CODE >= 23007,
              "SM120 MegaMoE needs NCCL 2.30.7 or newer (Device API: LSA + GIN)");

// Exactly one GIN backend: shim/cake_jit.py GIN_BACKENDS switches the others off by name. A backend
// it does not name (one a newer NCCL adds) would otherwise be inlined into every GIN call behind a
// runtime switch and grow the kernel's stack; GIN_BACKENDS then needs that backend's
// NCCL_GIN_<NAME>_ENABLE=0.
static_assert(NCCL_GIN_BACKEND_MASK_ALL != 0u &&
                  (NCCL_GIN_BACKEND_MASK_ALL & (NCCL_GIN_BACKEND_MASK_ALL - 1u)) == 0u,
              "SM120 MegaMoE compiles exactly one NCCL GIN backend; this NCCL has one that "
              "GIN_BACKENDS does not switch off");

#endif  // defined(__CUDA_ARCH__)
