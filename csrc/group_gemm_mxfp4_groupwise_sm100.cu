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
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/gemm/group_gemm_mxfp4_groupwise_sm100.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

#define DISPATCH_MMA_SM(mma_sm, MMA_SM, ...)  \
  [&]() -> bool {                             \
    if (mma_sm == 1) {                        \
      constexpr int MMA_SM = 1;               \
      return __VA_ARGS__();                   \
    } else if (mma_sm == 2) {                 \
      constexpr int MMA_SM = 2;               \
      return __VA_ARGS__();                   \
    }                                         \
    TORCH_CHECK(false, "Unsupported MMA SM"); \
    return false;                             \
  }()

#define DISPATCH_TILE_M(tile_m, TILE_M, ...)  \
  [&]() -> bool {                             \
    if (tile_m == 128) {                      \
      constexpr int TILE_M = 128;             \
      return __VA_ARGS__();                   \
    }                                         \
    TORCH_CHECK(false, "Unsupported TILE M"); \
    return false;                             \
  }()

#define DISPATCH_TILE_N(tile_n, TILE_N, ...)  \
  [&]() -> bool {                             \
    if (tile_n == 64) {                       \
      constexpr int TILE_N = 64;              \
      return __VA_ARGS__();                   \
    } else if (tile_n == 128) {               \
      constexpr int TILE_N = 128;             \
      return __VA_ARGS__();                   \
    } else if (tile_n == 192) {               \
      constexpr int TILE_N = 192;             \
      return __VA_ARGS__();                   \
    } else if (tile_n == 256) {               \
      constexpr int TILE_N = 256;             \
      return __VA_ARGS__();                   \
    }                                         \
    TORCH_CHECK(false, "Unsupported TILE N"); \
    return false;                             \
  }()

#define DISPATCH_TILE_K(tile_k, TILE_K, ...)  \
  [&]() -> bool {                             \
    if (tile_k == 128) {                      \
      constexpr int TILE_K = 128;             \
      return __VA_ARGS__();                   \
    } else if (tile_k == 256) {               \
      constexpr int TILE_K = 256;             \
      return __VA_ARGS__();                   \
    }                                         \
    TORCH_CHECK(false, "Unsupported TILE K"); \
    return false;                             \
  }()

#define DISPATCH_SWAP_AB(swap_ab, SWAP_AB, ...) \
  [&]() -> bool {                               \
    if (swap_ab == true) {                      \
      constexpr bool SWAP_AB = true;            \
      return __VA_ARGS__();                     \
    } else if (swap_ab == false) {              \
      constexpr bool SWAP_AB = false;           \
      return __VA_ARGS__();                     \
    }                                           \
    TORCH_CHECK(false, "Unsupported SWAP AB");  \
    return false;                               \
  }()

#define DISPATCH_PYTORCH_INPUT_OUTPUT_DTYPE(input_a_dtype, input_b_dtype, sf_a_dtype, sf_b_dtype, \
                                            output_dtype, c_type_in_a, c_type_in_b, c_type_sf_a,  \
                                            c_type_sf_b, c_type_out, ...)                         \
  [&]() -> bool {                                                                                 \
    return DISPATCH_PYTORCH_DTYPE_TO_CTYPE(output_dtype, c_type_out, [&] {                        \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_SF(sf_b_dtype, c_type_sf_b, [&] {                    \
        return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_SF(sf_a_dtype, c_type_sf_a, [&] {                  \
          return DISPATCH_PYTORCH_DTYPE_TO_CTYPE(input_b_dtype, c_type_in_b, [&] {                \
            return DISPATCH_PYTORCH_DTYPE_TO_CTYPE(input_a_dtype, c_type_in_a,                    \
                                                   [&] { return __VA_ARGS__(); });                \
          });                                                                                     \
        });                                                                                       \
      });                                                                                         \
    });                                                                                           \
  }()

template <typename T_A, typename T_B, typename T_SFA, typename T_SFB, typename T_OUT>
constexpr bool is_valid_config() {
  if constexpr ((std::is_same_v<T_A, __nv_fp8_e4m3> || std::is_same_v<T_A, __nv_fp8_e5m2>) &&
                std::is_same_v<T_B, __nv_fp4_e2m1> && std::is_same_v<T_SFA, __nv_fp8_e8m0> &&
                std::is_same_v<T_SFB, __nv_fp8_e8m0> &&
                (std::is_same_v<T_OUT, nv_half> || std::is_same_v<T_OUT, nv_bfloat16>)) {
    return true;
  }
  return false;
}

void CutlassGroupGemmMXFP4GroupwiseScaledSM100(at::Tensor int_workspace_buffer,
                                               at::Tensor float_workspace_buffer, at::Tensor A,
                                               at::Tensor B, at::Tensor SFA, at::Tensor SFB,
                                               at::Tensor D, at::Tensor m_indptr, int64_t n,
                                               int64_t k, int64_t mma_sm, int64_t tile_m,
                                               int64_t tile_n, int64_t tile_k, bool swap_ab) {
  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  int num_groups = m_indptr.size(0) - 1;
  DISPATCH_PYTORCH_INPUT_OUTPUT_DTYPE(
      A.scalar_type(), B.scalar_type(), SFA.scalar_type(), SFB.scalar_type(), D.scalar_type(),
      c_type_in_a, c_type_in_b, c_type_sf_a, c_type_sf_b, c_type_out, [&] {
        return DISPATCH_MMA_SM(mma_sm, MMA_SM, [&] {
          return DISPATCH_TILE_M(tile_m, TILE_M, [&] {
            return DISPATCH_TILE_N(tile_n, TILE_N, [&] {
              return DISPATCH_TILE_K(tile_k, TILE_K, [&] {
                return DISPATCH_SWAP_AB(swap_ab, SWAP_AB, [&] {
                  if constexpr (is_valid_config<c_type_in_a, c_type_in_b, c_type_sf_a, c_type_sf_b,
                                                c_type_out>()) {
                    using cutlass_t_in_a = cutlass_dtype_t<c_type_in_a>;
                    using cutlass_t_in_b = cutlass_dtype_t<c_type_in_b>;
                    using cutlass_t_sf_a = cutlass_dtype_t<c_type_sf_a>;
                    using cutlass_t_sf_b = cutlass_dtype_t<c_type_sf_b>;
                    using cutlass_t_out = cutlass_dtype_t<c_type_out>;
                    auto status = flashinfer::gemm::CutlassMXFP4GroupwiseScaledGroupGEMMSM100<
                        MMA_SM, TILE_M, TILE_N, TILE_K, SWAP_AB>(
                        static_cast<int*>(int_workspace_buffer.data_ptr()),
                        int_workspace_buffer.element_size() * int_workspace_buffer.size(0),
                        static_cast<float*>(float_workspace_buffer.data_ptr()),
                        float_workspace_buffer.element_size() * float_workspace_buffer.size(0),
                        static_cast<cutlass_t_in_a*>(A.data_ptr()),
                        static_cast<cutlass_t_in_b*>(B.data_ptr()),
                        static_cast<cutlass_t_sf_a*>(SFA.data_ptr()),
                        static_cast<cutlass_t_sf_b*>(SFB.data_ptr()),
                        static_cast<cutlass_t_out*>(D.data_ptr()),
                        static_cast<int*>(m_indptr.data_ptr()), n, k, num_groups, stream);
                    return status == cudaSuccess;
                  } else {
                    TORCH_CHECK(false, "Unsupported input data type");
                    return false;
                  }
                });
              });
            });
          });
        });
      });
}
