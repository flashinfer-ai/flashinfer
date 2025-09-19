#include <dlpack/dlpack.h>

#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/gemm/group_gemm_fp8_groupwise_sm100.cuh>

#include "tvm_binding_utils.h"

__global__ void simple_print_kernel(void* data, int dtype_code) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (dtype_code == kDLBfloat) {
      // bfloat16
      uint16_t* bf16_data = static_cast<uint16_t*>(data);
      uint32_t full = ((uint32_t)bf16_data[0]) << 16;
      float val = *reinterpret_cast<float*>(&full);
      printf("GPU: D[0] = %.6f\n", val);
    } else {
      // float32
      float* f32_data = static_cast<float*>(data);
      printf("GPU: D[0] = %.6f\n", f32_data[0]);
    }
  }
}

// following MACROS duplicates from flashinfer/csrc/group_gemm_fp8_groupwise_sm100.cu
#define DISPATCH_TVM_DTYPE_TO_CTYPE(tvm_dtype_in, tvm_dtype_out, c_type_in, c_type_out, ...) \
  [&]() -> bool {                                                                            \
    if (tvm_dtype_in.code == kDLFloat8_e4m3fn && tvm_dtype_in.bits == 8) {                   \
      using c_type_in = cutlass::float_e4m3_t;                                               \
      if (tvm_dtype_out.code == kDLFloat && tvm_dtype_out.bits == 16) {                      \
        using c_type_out = cutlass::half_t;                                                  \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
      if (tvm_dtype_out.code == kDLBfloat && tvm_dtype_out.bits == 16) {                     \
        using c_type_out = cutlass::bfloat16_t;                                              \
        return __VA_ARGS__();                                                                \
      }                                                                                      \
    }                                                                                        \
    CHECK(false) << "Unsupported TVM dtype combination: input(" << tvm_dtype_in.code << ","  \
                 << tvm_dtype_in.bits << ") output(" << tvm_dtype_out.code << ","            \
                 << tvm_dtype_out.bits << ")";                                               \
    return false;                                                                            \
  }()

#define DISPATCH_MMA_SM(mma_sm, MMA_SM, ...)          \
  [&]() -> bool {                                     \
    if (mma_sm == 1) {                                \
      constexpr int MMA_SM = 1;                       \
      return __VA_ARGS__();                           \
    } else if (mma_sm == 2) {                         \
      constexpr int MMA_SM = 2;                       \
      return __VA_ARGS__();                           \
    }                                                 \
    CHECK(false) << "Unsupported MMA SM: " << mma_sm; \
    return false;                                     \
  }()

#define DISPATCH_SCALE_GRANULARITY(scale_granularity_m, scale_granularity_n, scale_granularity_k, \
                                   SCALE_GRANULARITY_M, SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, \
                                   ...)                                                           \
  [&]() -> bool {                                                                                 \
    if (scale_granularity_m == 1 && scale_granularity_n == 128 && scale_granularity_k == 128) {   \
      constexpr int SCALE_GRANULARITY_M = 1;                                                      \
      constexpr int SCALE_GRANULARITY_N = 128;                                                    \
      constexpr int SCALE_GRANULARITY_K = 128;                                                    \
      return __VA_ARGS__();                                                                       \
    } else if (scale_granularity_m == 128 && scale_granularity_n == 128 &&                        \
               scale_granularity_k == 128) {                                                      \
      constexpr int SCALE_GRANULARITY_M = 128;                                                    \
      constexpr int SCALE_GRANULARITY_N = 128;                                                    \
      constexpr int SCALE_GRANULARITY_K = 128;                                                    \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    CHECK(false) << "Unsupported scale granularity: (" << scale_granularity_m << ","              \
                 << scale_granularity_n << "," << scale_granularity_k << ")";                     \
    return false;                                                                                 \
  }()

#define DISPATCH_SCALE_MAJOR_K(scale_major_mode, SCALE_MAJOR_K, ...)      \
  [&]() -> bool {                                                         \
    if (scale_major_mode == 0) {                                          \
      constexpr bool SCALE_MAJOR_K = true;                                \
      return __VA_ARGS__();                                               \
    } else if (scale_major_mode == 1) {                                   \
      constexpr bool SCALE_MAJOR_K = false;                               \
      return __VA_ARGS__();                                               \
    }                                                                     \
    CHECK(false) << "Unsupported Scale Major Mode: " << scale_major_mode; \
    return false;                                                         \
  }()

namespace flashinfer {
namespace group_gemm {

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          int MmaSM, typename DTypeIn, typename DTypeOut>
cudaError_t CutlassFP8GroupwiseScaledGroupGEMMSM100(
    void* int_buffer, size_t int_buffer_size_in_bytes, void* float_buffer,
    size_t float_buffer_size_in_bytes, DTypeIn* A, DTypeIn* B, float* SFA, float* SFB, DTypeOut* D,
    int* m_indptr, int max_m, int n, int k, int num_groups, cudaStream_t stream);

}  // namespace group_gemm
}  // namespace flashinfer

// FP8 Group GEMM implementation with CUTLASS for SM100A (Blackwell)
int GroupedGemmFp8Run(DLTensor* int_workspace_buffer, DLTensor* float_workspace_buffer, DLTensor* A,
                      DLTensor* B, DLTensor* SFA, DLTensor* SFB, DLTensor* D, DLTensor* m_indptr,
                      int64_t n, int64_t k, int64_t scale_granularity_m,
                      int64_t scale_granularity_n, int64_t scale_granularity_k,
                      int64_t scale_major_mode, int64_t mma_sm, TVMStreamHandle cuda_stream) {
  LOG(INFO) << "Call to GroupedGemmFp8Run";

  if (!int_workspace_buffer || !float_workspace_buffer) {
    LOG(FATAL) << "work space buffer is null";
  }
  if (!A || !B) {
    LOG(FATAL) << "A or B is null";
  }
  if (!SFA || !SFB) {
    LOG(FATAL) << "Scale for A or B is null";
  }
  if (!D) {
    LOG(FATAL) << "D is null";
  }
  if (!m_indptr) {
    LOG(FATAL) << "m indptr is null";
  }

  LOG(INFO) << "=== NULL CHECKS PASSED ===";

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  size_t float_workspace_size =
      float_workspace_buffer->shape[0] * DataType(float_workspace_buffer->dtype).bytes();
  size_t int_workspace_size =
      int_workspace_buffer->shape[0] * DataType(int_workspace_buffer->dtype).bytes();

  int64_t num_groups = m_indptr->shape[0] - 1;
  int64_t max_m = SFA->shape[1];

  try {
    LOG(INFO) << "=== STARTING DISPATCH ===";
    DISPATCH_TVM_DTYPE_TO_CTYPE(A->dtype, D->dtype, c_type_in, c_type_out, [&] {
      LOG(INFO) << "=== INSIDE DTYPE DISPATCH ===";
      return DISPATCH_SCALE_MAJOR_K(scale_major_mode, SCALE_MAJOR_K, [&] {
        LOG(INFO) << "=== INSIDE SCALE_MAJOR_K DISPATCH ===";
        return DISPATCH_MMA_SM(mma_sm, MMA_SM, [&] {
          LOG(INFO) << "=== INSIDE MMA_SM DISPATCH ===";
          return DISPATCH_SCALE_GRANULARITY(
              scale_granularity_m, scale_granularity_n, scale_granularity_k, SCALE_GRANULARITY_M,
              SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, [&] {
                // Validate kernel parameters before call
                LOG(INFO) << "Kernel parameters:";
                LOG(INFO) << "  max_m: " << max_m;
                LOG(INFO) << "  n: " << n;
                LOG(INFO) << "  k: " << k;
                LOG(INFO) << "  num_groups: " << num_groups;
                LOG(INFO) << "  scale_granularity: (" << scale_granularity_m << ","
                          << scale_granularity_n << "," << scale_granularity_k << ")";
                LOG(INFO) << "  scale_major_mode: " << scale_major_mode;
                LOG(INFO) << "  mma_sm: " << mma_sm;

                using cutlass_t_in = flashinfer::cutlass_dtype_t<c_type_in>;
                using cutlass_t_out = flashinfer::cutlass_dtype_t<c_type_out>;
                LOG(INFO) << "CUTLASS types created successfully";
                LOG(INFO) << "sizeof(cutlass_t_in): " << sizeof(cutlass_t_in);
                LOG(INFO) << "sizeof(cutlass_t_out): " << sizeof(cutlass_t_out);

                LOG(INFO) << "Calling actual kernel now...";

                auto status = flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM100<
                    SCALE_GRANULARITY_M, SCALE_GRANULARITY_N, SCALE_GRANULARITY_K, SCALE_MAJOR_K,
                    MMA_SM>(
                    static_cast<int32_t*>(int_workspace_buffer->data) +
                        int_workspace_buffer->byte_offset / sizeof(int32_t),
                    int_workspace_buffer->shape[0] * sizeof(int32_t),
                    static_cast<float*>(float_workspace_buffer->data) +
                        float_workspace_buffer->byte_offset / sizeof(float),
                    float_workspace_buffer->shape[0] * sizeof(float),
                    static_cast<cutlass_t_in*>(A->data) + A->byte_offset / sizeof(cutlass_t_in),
                    static_cast<cutlass_t_in*>(B->data) + B->byte_offset / sizeof(cutlass_t_in),
                    static_cast<float*>(SFA->data) + SFA->byte_offset / sizeof(float),
                    static_cast<float*>(SFB->data) + SFB->byte_offset / sizeof(float),
                    static_cast<cutlass_t_out*>(D->data) + D->byte_offset / sizeof(cutlass_t_out),
                    static_cast<int32_t*>(m_indptr->data) + m_indptr->byte_offset / sizeof(int32_t),
                    max_m, n, k, num_groups, stream);

                // Check for CUDA errors immediately after kernel call
                cudaError_t cuda_error = cudaGetLastError();
                if (cuda_error != cudaSuccess) {
                  LOG(ERROR) << "CUDA error after kernel call: " << cudaGetErrorString(cuda_error);
                  return false;
                }
                simple_print_kernel<<<1, 1>>>(static_cast<char*>(D->data) + D->byte_offset,
                                              D->dtype.code);

                LOG(INFO) << "Kernel execution completed successfully";

                return status == cudaSuccess;
              });
        });
      });
    })
    ? 0 : -3;

  } catch (const std::exception& e) {
    LOG(INFO) << "Exception caught:" << e.what();
    return -4;
  }
}
