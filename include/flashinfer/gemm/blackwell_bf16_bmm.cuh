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

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(symbol)                               \
  extern "C" __global__ void symbol(__nv_bfloat16* A, __nv_bfloat16* B_storage,       \
                                    uint8_t* out_bytes, int M, int N, int a_stride_b, \
                                    int a_stride_m, int a_stride_k, int b_stride_b,   \
                                    int b_stride_n, int b_stride_k, int out_type)

FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k64);
FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256);
FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024);

FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_m32n40_o0_fixed);
FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_m32n40_o1_fixed);
FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_m32n40_o2_fixed);

FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_full_m128n64o0_fixed);
FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_full_m128n64o1_fixed);
FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k256_full_m128n64o2_fixed);

FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_full_m16n1024o0_fixed);
FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_full_m16n1024o1_fixed);
FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_full_m16n1024o2_fixed);

FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL(
    kernel_flashinfer_blackwell_bf16_bmm_goal_dispatcher_v1_k1024_n16_m8_tail);

#undef FLASHINFER_DECLARE_CAKE_BF16_BMM_KERNEL
