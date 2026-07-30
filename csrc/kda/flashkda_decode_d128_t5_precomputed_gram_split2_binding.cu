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

#include "flashkda_decode_binding_common.cuh"

#define FLASHKDA_DECODE_VALUE_SPLIT 2
#define FLASHKDA_DECODE_LAUNCH_THREADS 160
#define FLASHKDA_DECODE_EXPECTED_SMEM 31360

#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
#define LoomTensorMap flashkda_generated_LoomTensorMap
#define LoomTensorMapPack flashkda_generated_LoomTensorMapPack
#define CUtensorMap flashkda_generated_CUtensorMap
#include "flashkda_decode_d128_t5_precomputed_gram_split2.cu"
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t
#undef LoomTensorMap
#undef LoomTensorMapPack
#undef CUtensorMap

#include "flashkda_decode_binding_impl.cuh"
