/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define FLASHINFER_CAKE_KDA_AFFINE_BODY_FILE "cake_kda_affine_unbounded_softplus_main.cu"
#define FLASHINFER_CAKE_KDA_AFFINE_KERNEL kernel_cake_kda_affine_unbounded_softplus_main
#define FLASHINFER_CAKE_KDA_AFFINE_THREADS 1024
#define FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES 227968
#define FLASHINFER_CAKE_KDA_AFFINE_USE_PDL 0
#define FLASHINFER_CAKE_KDA_AFFINE_ROLE FLASHINFER_CAKE_KDA_AFFINE_ROLE_MAIN
#define FLASHINFER_CAKE_KDA_AFFINE_ARG_PLAN_SHA256 "22022915599b937c578a5a511920e0b48fd693aa0f414df0c1eb1a0684ad97e2"
#include "cake_kda_affine_direct_m128_binding.cuh"
