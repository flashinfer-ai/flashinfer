/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <flashinfer/attention/blackwell/fmha_cutlass_sm100.cuh>
#include <flashinfer/cutlass_utils.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

#define DISPATCH_DTYPE_IN_OUT(in_dtype, out_dtype, c_type_in, c_type_out, ...)      \
  [&]() -> bool {                                                                   \
    if (in_dtype == out_dtype) {                                                    \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(in_dtype, c_type_in, [&] {        \
        using c_type_out = c_type_in;                                               \
        return __VA_ARGS__();                                                       \
      });                                                                           \
    } else {                                                                        \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(in_dtype, c_type_in, [&] {         \
        return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(out_dtype, c_type_out,          \
                                                    [&] { return __VA_ARGS__(); }); \
      });                                                                           \
    }                                                                               \
  }()

void FMHACutlassSM100Run(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor qo_lens,
                         at::Tensor kv_lens, at::Tensor qo_segment_offsets,
                         at::Tensor kv_segment_offsets, at::Tensor o,
                         std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                         double sm_scale, int64_t num_qo_heads, int64_t num_kv_heads,
                         int64_t head_dim_qk, int64_t head_dim_vo, int64_t batch_size,
                         int64_t total_qo_len, int64_t total_kv_len, int64_t max_qo_len,
                         int64_t max_kv_len) {
  CHECK(q.scalar_type() == k.scalar_type());
  auto scalar_type_in = q.scalar_type();
  auto scalar_type_out = o.scalar_type();
  DISPATCH_DTYPE_IN_OUT(scalar_type_in, scalar_type_out, c_type_in, c_type_out, [&] {
    using cutlass_type_in = cutlass_dtype_t<c_type_in>;
    using cutlass_type_out = cutlass_dtype_t<c_type_out>;
    using TILE_Q = _256;
    using TILE_KV = _128;
    using D_QK = _192;
    using D_VO = _128;
    using TileShapeQK = Shape<TILE_Q, TILE_KV, D_QK>;
    using TileShapePV = Shape<TILE_Q, D_VO, TILE_KV>;
    if (mask_mode_code == 1) {
      FwdRunner<cutlass_type_in, cutlass_type_out, TileShapeQK, TileShapePV, void, CausalMask>
          runner;
      runner.run(q, k, v, qo_lens, kv_lens, qo_segment_offsets, kv_segment_offsets, o, maybe_lse,
                 mask_mode_code, sm_scale, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
                 batch_size, total_qo_len, total_kv_len, max_qo_len, max_kv_len);

    } else {
      FwdRunner<cutlass_type_in, cutlass_type_out, TileShapeQK, TileShapePV, void, ResidualMask>
          runner;
      runner.run(q, k, v, qo_lens, kv_lens, qo_segment_offsets, kv_segment_offsets, o, maybe_lse,
                 mask_mode_code, sm_scale, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
                 batch_size, total_qo_len, total_kv_len, max_qo_len, max_kv_len);
    }
    return true;
  });
}
