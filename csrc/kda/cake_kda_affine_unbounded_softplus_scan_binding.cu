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

#define FLASHINFER_CAKE_KDA_AFFINE_BODY_FILE "cake_kda_affine_unbounded_softplus_scan.cu"
#define FLASHINFER_CAKE_KDA_AFFINE_KERNEL kernel_cake_kda_affine_unbounded_softplus_scan
#define FLASHINFER_CAKE_KDA_AFFINE_THREADS 128
#define FLASHINFER_CAKE_KDA_AFFINE_SMEM_BYTES 66560
#define FLASHINFER_CAKE_KDA_AFFINE_USE_PDL 1
#define FLASHINFER_CAKE_KDA_AFFINE_ROLE FLASHINFER_CAKE_KDA_AFFINE_ROLE_SCAN
#define FLASHINFER_CAKE_KDA_AFFINE_ARG_PLAN_SHA256 \
  "a9a2d480c8bfe74cc155384a93039ad82c481c5e00f4f40b800b367aa388474b"
#include "cake_kda_affine_scan_binding.cuh"
