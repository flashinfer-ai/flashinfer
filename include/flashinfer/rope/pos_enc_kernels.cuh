/*
 * Copyright (c) 2023-2026 by FlashInfer team.
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
#ifndef FLASHINFER_ROPE_KERNELS_CUDA_POS_ENC_KERNELS_CUH_
#define FLASHINFER_ROPE_KERNELS_CUDA_POS_ENC_KERNELS_CUH_

/*
 * Standalone RoPE Kernels - Unified Header
 * =========================================
 *
 * This header provides a unified include for the standalone RoPE API kernels
 * (apply_rope, apply_llama31_rope, rope_quantize, etc.):
 *
 * - types.cuh: RoPE-specific parameter structs
 * - kernels.cuh: CUDA kernel implementations
 * - launchers.cuh: Host launcher functions with dispatch logic
 *
 * For attention-level positional encoding (PosEncodingMode, inline RoPE helpers),
 * see include/flashinfer/pos_enc.cuh.
 */

#include "flashinfer/rope/kernels.cuh"
#include "flashinfer/rope/launchers.cuh"
#include "flashinfer/rope/types.cuh"

#endif  // FLASHINFER_ROPE_KERNELS_CUDA_POS_ENC_KERNELS_CUH_
