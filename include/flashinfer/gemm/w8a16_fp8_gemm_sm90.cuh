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
#ifndef FLASHINFER_GEMM_W8A16_FP8_GEMM_SM90_CUH_
#define FLASHINFER_GEMM_W8A16_FP8_GEMM_SM90_CUH_

// W8A16 GEMM for Hopper (sm_90a) on CUTLASS's mixed-input mainloop.
//
//   out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])
//
// The pieces, in include order (w8a16_sm90/; shared with Blackwell: w8a16_common.hpp,
// w8a16_convert.hpp for the e4m3 -> bf16 / f16 fragment conversion that replaces CUTLASS's
// converter, w8a16_host.hpp for the host-side arguments / workspace / run sequence):
//   scheduler.hpp   the padding-free L2-grouped persistent tile scheduler
//   tile_order.hpp  raster order and L2 group size per problem
//   gemm.hpp        the tile configurations and the W8A16Sm90Gemm kernel variants
//   dispatch.hpp    `algo` codes, split-K slicing, the cost model, the variant table
// The launcher (csrc/w8a16_fp8_gemm.cu) calls W8A16Sm90Resolve, then W8A16Fp8GemmSm90 /
// W8A16Fp8GemmSm90WorkspaceBytes with the resulting plan.

#include "flashinfer/gemm/w8a16_sm90/dispatch.hpp"

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_SM90_CUH_
