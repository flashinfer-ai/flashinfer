#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

extern "C" void launch_fp8_gemm(const void* A, const void* B, void* Out, const void* A_scale,
                                const void* B_scale, int M, int N, int K, void* workspace,
                                int splits, cudaStream_t stream);

namespace {

int compute_splits(int M, int N, int K) {
  if (M <= 16 && N <= 64 && K <= 2048 && K >= 512 && (K % 16 == 0)) {
    return 1;
  }

  int BM, BN;
  const bool small_m = (M <= 16);
  if (small_m) {
    BM = 16;
    BN = 64;
  } else if (M <= 32) {
    BM = 32;
    BN = 128;
  } else if (M <= 64) {
    BM = 64;
    BN = (N <= 1024) ? 64 : 128;
  } else if (M <= 128) {
    BM = (N <= 4096) ? 64 : 128;
    BN = (N <= 1024) ? 64 : 128;
  } else if (M >= 1024) {
    BM = 192;
    BN = 128;
  } else {
    BM = (N <= 1024) ? 64 : 128;
    BN = (N <= 1024) ? 64 : 128;
  }
  const int blocks = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);

  bool split_ok = false;
  if (blocks < 8) {
    split_ok = (K >= 1024);
  } else if (blocks < 16) {
    split_ok = (K >= 2048);
  } else if (blocks < 24) {
    split_ok = small_m ? (K >= 2048) : (K >= 4096);
  } else if (blocks < 32) {
    split_ok = small_m ? (K >= 2048) : (K >= 4096);
  } else if (blocks < 48) {
    split_ok = (K >= 4096);
  } else if (blocks < 96) {
    split_ok = small_m && (K >= 4096);
  } else if (blocks < 128) {
    split_ok = small_m && (K >= 8192);
  }
  if (!split_ok) {
    return 1;
  }

  const int min_K_per_split = small_m ? ((blocks <= 16) ? 256 : (blocks <= 64 ? 512 : 1024)) : 1024;
  const int max_useful_splits = K / min_K_per_split;
  if (max_useful_splits < 2) {
    return 1;
  }

  const int target = small_m ? 256 : 96;
  int splits = (target + blocks - 1) / blocks;
  if (splits > max_useful_splits) {
    splits = max_useful_splits;
  }
  if (small_m && K >= 8192 && blocks >= 32 && splits > 3) {
    splits = 3;
  }
  if (small_m && K <= 4096 && splits > 4) {
    splits = 4;
  }
  if (splits > 8) {
    splits = 8;
  }
  return splits < 2 ? 1 : splits;
}

}  // namespace

void bmm_fp8_sm121_cuda_run(TensorView A, TensorView B, TensorView A_scale, TensorView B_scale,
                            TensorView out, TensorView workspace) {
  CHECK_INPUT_AND_TYPE(A, dl_float8_e4m3fn);
  CHECK_CUDA(B);
  CHECK_INPUT_TYPE(B, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_CUDA(A_scale);
  CHECK_CUDA(B_scale);
  CHECK_INPUT_TYPE(A_scale, dl_float32);
  CHECK_INPUT_TYPE(B_scale, dl_float32);
  CHECK_CUDA(workspace);
  CHECK_INPUT_TYPE(workspace, dl_float32);
  CHECK_DIM(3, A);
  CHECK_DIM(3, B);
  CHECK_DIM(3, out);
  CHECK_DEVICE(A, B);
  CHECK_DEVICE(A, A_scale);
  CHECK_DEVICE(A, B_scale);
  CHECK_DEVICE(A, out);
  CHECK_DEVICE(A, workspace);
  TVM_FFI_ICHECK_EQ(A_scale.numel(), 1) << "A_scale must have one element";
  TVM_FFI_ICHECK_EQ(B_scale.numel(), 1) << "B_scale must have one element";

  const int batch = static_cast<int>(A.size(0));
  const int M = static_cast<int>(A.size(1));
  const int K = static_cast<int>(A.size(2));
  const int N = static_cast<int>(B.size(2));
  TVM_FFI_ICHECK_EQ(B.size(0), batch) << "B batch dimension must match A";
  TVM_FFI_ICHECK_EQ(B.size(1), K) << "B K dimension must match A";
  TVM_FFI_ICHECK_EQ(out.size(0), batch) << "out batch dimension must match A";
  TVM_FFI_ICHECK_EQ(out.size(1), M) << "out M dimension must match A";
  TVM_FFI_ICHECK_EQ(out.size(2), N) << "out N dimension must match B";
  TVM_FFI_ICHECK_EQ(B.stride(1), 1) << "B must be a transposed column-major view";
  TVM_FFI_ICHECK_EQ(B.stride(2), K) << "B must be a transposed column-major view";

  const int splits = compute_splits(M, N, K);
  const int64_t required_workspace = splits > 1 ? static_cast<int64_t>(splits) * M * N : 0;
  TVM_FFI_ICHECK_GE(workspace.numel(), required_workspace)
      << "workspace is too small for bmm_fp8 specialized kernel";

  ffi::CUDADeviceGuard device_guard(A.device().device_id);
  cudaStream_t stream = get_stream(A.device());
  const int64_t A_batch_stride = static_cast<int64_t>(M) * K;
  const int64_t B_batch_stride = static_cast<int64_t>(K) * N;
  const int64_t O_batch_stride = static_cast<int64_t>(M) * N;
  void* workspace_ptr = required_workspace > 0 ? workspace.data_ptr() : nullptr;

  for (int b = 0; b < batch; ++b) {
    const void* Ap = static_cast<const uint8_t*>(A.data_ptr()) + b * A_batch_stride;
    const void* Bp = static_cast<const uint8_t*>(B.data_ptr()) + b * B_batch_stride;
    void* Op = static_cast<uint8_t*>(out.data_ptr()) + b * O_batch_stride * sizeof(__nv_bfloat16);
    launch_fp8_gemm(Ap, Bp, Op, A_scale.data_ptr(), B_scale.data_ptr(), M, N, K, workspace_ptr,
                    splits, stream);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, bmm_fp8_sm121_cuda_run);
