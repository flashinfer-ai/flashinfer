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
#ifndef FLASHINFER_GEMM_W8A16_FP8_GEMM_SM100_CUH_
#define FLASHINFER_GEMM_W8A16_FP8_GEMM_SM100_CUH_

// W8A16 GEMM for Blackwell (sm_100a): FlashInfer's fork of CUTLASS's mixed-input kernel
// with two transform warpgroups and a menu of tiles, picked per problem by a wave-fill cost
// model fitted to measurements on the Cosmos3-Super shapes (benchmarks/bench_w8a16_fp8_gemm.py).
//
//   out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])
//
// The pieces (w8a16_sm100/, shared parts in w8a16_common.hpp):
//   policy.hpp, builder.hpp, collective.hpp, kernel.hpp
//                   the forked CUTLASS mixed-input mainloop (transform warpgroups, A in
//                   TMEM or smem)
//   convert.hpp     e4m3 -> bf16 / f16 conversion the transform warps run
//   tile_order.hpp  tile-group size, raster order and TMA cache hints per problem
//   gemm.hpp        the tile configurations and the W8A16Sm100Gemm kernel wrapper
//   dispatch.hpp    `algo` codes, split-K / stream-K rules, the cost model, the tables
// The launcher (csrc/w8a16_fp8_gemm_sm100.cu) calls W8A16Sm100Resolve, then
// W8A16Fp8GemmSm100 / W8A16Fp8GemmSm100WorkspaceBytes with the resulting plan.

#include "flashinfer/gemm/w8a16_sm100/dispatch.hpp"

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_SM100_CUH_
