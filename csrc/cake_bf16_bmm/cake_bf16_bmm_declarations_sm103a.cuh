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

extern "C" __global__ void kernel_cake_bf16_bmm_05210b13930b347b10e5(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_0aa6610c5790bb7e78a8(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_1355ce28cd6e7bb672aa(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_16a6b9251e408d683478(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_20eb25725fe130422815(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_214585a1ff53e7aa90e4(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_2be765abed4a60c000a7(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_2d6361a99d62f3f52a21(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_3c2820b9896ee6f117ab(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_40eba7ca5a71bfe0c7b8(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_44eb4a7b8a86325202c6(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_4c2925f941c18e9ba383(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_4c33ad0ae6c78c36746c(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_59b89d08f864e7dfae86(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_5a2711b7cb94fd49c706(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_706b766a7aedb03057b4(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_78ff57242075aae196f4(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_869f43afdb8a948398b7(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_9589f85cc4b9aafde12a(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_97ac58f8c28e3a9f7ccd(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_9a3f18b71438e9f341b6(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_a13456761a57e20eb43d(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_a1dc7b1314ac48d86dbf(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_af7495a29467dba4178c(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_b051749a89c3c70704d1(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_b751bc0ab613774d2a8d(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_b7b3dfc29cbe1e75cea8(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_be0a25527b6a907a6b9f(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_c23b198fd16a09495a3c(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_c76c2166a5d5c53f4d72(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_dade0fe0e8e644a42572(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_dfb467d2ac52721f071a(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_e16dc73fa76da4f687c4(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);

extern "C" __global__ void kernel_cake_bf16_bmm_fdbd1b90786b149d7d2e(__nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ B_storage, uint8_t* __restrict__ out_bytes, int M, int N, int a_stride_b, int a_stride_m, int a_stride_k, int b_stride_b, int b_stride_n, int b_stride_k, int out_type);
