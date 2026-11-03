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

extern "C" __global__ void kernel_cake_bf16_bmm_007557bb28b86b0ad447(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_1428cc8db4c88a1a4faf(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_148ff052ab7f16459f74(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_215b1124d8a15af5cdd3(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_49bbceba1010338dc287(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_5143d7dde41d5c40f467(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_5ca33069bbbad5c7016d(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_5fb134eefb0c3d4d3572(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_60c16df3a3f9b440c477(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_6bb5fd7ca479b4132d28(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_6d8fa1a07f5f07cab697(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_73c9472b3d1b2ce56afe(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_74b735e0c2a4a0ab11b5(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_75d6b3b338da1b83fdd2(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_82a7d836ee60f3f8e54d(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_86f88c11bbab28a112a4(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_8bbbf16d966b721a5216(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_971934c57c7ec8e277e8(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_9fc7d7b7d693e3601c8e(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_a620001ffb4be97be504(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_aae1fbf4100e7044666d(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_b3614c1b24fcc0d9db41(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_b45cfbc87416c25a8d38(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_bc0d392cafbe0e91f5aa(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_bda02d92fb3e9e813ac2(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_beecb1dd16619150e5dc(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_cb886960b55646251e65(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_cc44de244d2ddb40e3e0(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_cd337271d0ea1ad303f9(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_d1f4a5edaf1b6f5d5e54(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_d53a138927da127b43f3(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_daddfc793176eeaf96ba(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_dd1aa52c1efb755d163e(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_ed865140f48d71197566(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);
