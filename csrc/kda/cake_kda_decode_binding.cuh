/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#ifndef CAKE_KDA_DECODE_BODY_FILE
#error "CAKE_KDA_DECODE_BODY_FILE must be defined by the binding translation unit"
#endif

#include "cake_kda_decode_binding_common.cuh"

// Frozen Loom bodies provide private fixed-width aliases. Rename them while
// including the body so CUDA's standard integer typedefs remain unambiguous in
// the TVM-FFI binding.
#define uint8_t cake_kda_generated_uint8_t
#define uint16_t cake_kda_generated_uint16_t
#define uint32_t cake_kda_generated_uint32_t
#define uint64_t cake_kda_generated_uint64_t
#define int32_t cake_kda_generated_int32_t
#define int16_t cake_kda_generated_int16_t
#define LoomTensorMap cake_kda_generated_LoomTensorMap
#define LoomTensorMapPack cake_kda_generated_LoomTensorMapPack
#define CUtensorMap cake_kda_generated_CUtensorMap
#include CAKE_KDA_DECODE_BODY_FILE
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t
#undef LoomTensorMap
#undef LoomTensorMapPack
#undef CUtensorMap

#ifdef CAKE_KDA_DECODE_DIRECT_IMPL
#include "cake_kda_decode_binding_direct_impl.cuh"
#else
#include "cake_kda_decode_binding_impl.cuh"
#endif
