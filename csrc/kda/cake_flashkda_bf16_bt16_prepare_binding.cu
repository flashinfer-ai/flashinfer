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

#include "cake_flashkda_bt16_binding_common.cuh"

#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
#define CakeTensorMap flashkda_generated_CakeTensorMap
#define CakeTensorMapPack flashkda_generated_CakeTensorMapPack
#define CUtensorMap flashkda_generated_CUtensorMap
#include "cake_flashkda_bf16_bt16_prepare.cu"
#undef CUtensorMap
#undef CakeTensorMapPack
#undef CakeTensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

#define FLASHINFER_BT16_PREPARE_KERNEL kernel_flashkda_bf16_bt16_prepare
#define FLASHINFER_BT16_PREPARE_USES_BETA_TMA 0
#include "cake_flashkda_bt16_prepare_binding_impl.cuh"
