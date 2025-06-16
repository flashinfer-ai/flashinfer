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
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/cutlass_utils.cuh>

#include "pytorch_extension_utils.h"

#define DISPATCH_mask_mode(mask_mode, MASK_MODE, ...)   \
  [&]() -> bool {                                       \
    if (mask_mode == MaskMode::kNone) {                 \
      constexpr MaskMode MASK_MODE = MaskMode::kNone;   \
      return __VA_ARGS__();                             \
    } else if (mask_mode == MaskMode::kCausal) {        \
      constexpr MaskMode MASK_MODE = MaskMode::kCausal; \
      return __VA_ARGS__();                             \
    }                                                   \
    return false;                                       \
  }()

#define DISPATCH_head_dim(head_dim_qk, head_dim_vo, HEAD_DIM_QK, HEAD_DIM_VO, ...) \
  [&]() -> bool {                                                                  \
    if (head_dim_qk == 192 && head_dim_vo == 128) {                                \
      constexpr int HEAD_DIM_QK = 192;                                             \
      constexpr int HEAD_DIM_VO = 128;                                             \
      return __VA_ARGS__();                                                        \
    } else if (head_dim_qk == 128 && head_dim_vo == 128) {                         \
      constexpr int HEAD_DIM_QK = 128;                                             \
      constexpr int HEAD_DIM_VO = 128;                                             \
      return __VA_ARGS__();                                                        \
    }                                                                              \
    return false;                                                                  \
  }()

#define DISPATCH_DTYPE_IN_OUT(in_dtype, out_dtype, c_type_in, c_type_out, ...) \
  [&]() -> bool {                                                              \
    if (in_dtype == out_dtype) {                                               \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(in_dtype, c_type_in, [&] {   \
        using c_type_out = c_type_in;                                          \
        return __VA_ARGS__();                                                  \
      });                                                                      \
    }                                                                          \
    return false;                                                              \
  }()

#define DISPATCH_context(DTypeIn, DTypeOut, HEAD_DIM_QK, HEAD_DIM_VO, MaskMode, ...)         \
  {                                                                                          \
    DISPATCH_mask_mode(mask_mode, MaskMode, [&] {                                            \
      return DISPATCH_DTYPE_IN_OUT(scalar_type_in, scalar_type_out, DTypeIn, DTypeOut, [&] { \
        return DISPATCH_head_dim(head_dim_qk, head_dim_vo, HEAD_DIM_QK, HEAD_DIM_VO,         \
                                 [&] { return __VA_ARGS__(); });                             \
      });                                                                                    \
    });                                                                                      \
  }

using namespace flashinfer;

void FMHACutlassSM100Run(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k, at::Tensor v,
                         at::Tensor qo_segment_offsets, at::Tensor kv_segment_offsets,
                         at::Tensor work_indptr, at::Tensor qo_tile_indices,
                         at::Tensor qo_head_indices, at::Tensor batch_indices, at::Tensor o,
                         std::optional<at::Tensor> maybe_lse, int64_t mask_mode_code,
                         double sm_scale, int64_t num_qo_heads, int64_t num_kv_heads,
                         int64_t head_dim_qk, int64_t head_dim_vo, int64_t max_qo_len) {
  CHECK(q.scalar_type() == k.scalar_type());
  auto scalar_type_in = q.scalar_type();
  auto scalar_type_out = o.scalar_type();
  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  int total_qo_len = q.size(0);
  int total_kv_len = k.size(0);
  int batch_size = qo_segment_offsets.size(0) - 1;
  int q_stride_n = q.stride(0);
  int q_stride_h = q.stride(1);
  int k_stride_n = k.stride(0);
  int k_stride_h = k.stride(1);
  int v_stride_n = v.stride(0);
  int v_stride_h = v.stride(1);

  const c10::cuda::OptionalCUDAGuard device_guard(qo_segment_offsets.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_context(DTypeIn, DTypeOut, HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, [&] {
    using cutlass_type_in = cutlass_dtype_t<DTypeIn>;
    using cutlass_type_out = cutlass_dtype_t<DTypeOut>;
    using TILE_Q = _256;
    using TILE_KV = _128;
    using D_QK = cute::Int<HEAD_DIM_QK>;
    using D_VO = cute::Int<HEAD_DIM_VO>;
    using TileShapeQK = Shape<TILE_Q, TILE_KV, D_QK>;
    using TileShapePV = Shape<TILE_Q, D_VO, TILE_KV>;
    using CutlassMaskMode =
        typename std::conditional<MASK_MODE == MaskMode::kCausal, CausalMask, ResidualMask>::type;
    auto status = run_fmha_fwd<cutlass_type_in, cutlass_type_out, int32_t, TileShapeQK, TileShapePV,
                               CutlassMaskMode>(
        workspace_buffer.data_ptr(), static_cast<cutlass_type_in*>(q.data_ptr()),
        static_cast<cutlass_type_in*>(k.data_ptr()), static_cast<cutlass_type_in*>(v.data_ptr()),
        static_cast<int*>(qo_segment_offsets.data_ptr()),
        static_cast<int*>(kv_segment_offsets.data_ptr()), static_cast<int*>(work_indptr.data_ptr()),
        static_cast<int*>(qo_tile_indices.data_ptr()),
        static_cast<int*>(qo_head_indices.data_ptr()), static_cast<int*>(batch_indices.data_ptr()),
        static_cast<cutlass_type_out*>(o.data_ptr()),
        maybe_lse.has_value() ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr,
        mask_mode_code, sm_scale, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, q_stride_n,
        q_stride_h, k_stride_n, k_stride_h, v_stride_n, v_stride_h, batch_size, total_qo_len,
        total_kv_len, max_qo_len, stream);
    TORCH_CHECK(status == cudaSuccess, "Cutlass FMHA forward pass failed",
                cudaGetErrorString(status));

    return true;
  });
}
