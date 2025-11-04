#include "flashinfer/gemm/dsv3_router_gemm.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::trtllm_dsv3_router_gemm {
template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemm(float* output, T const* mat_a, T const* mat_b, cudaStream_t stream,
                      bool use_pdl = false) {
  constexpr int VPT = 16 / sizeof(T);
  constexpr int kBlockSize = 128;
  cudaLaunchConfig_t config;
  config.gridDim = kNumExperts;
  config.blockDim = kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = use_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  auto status = cudaLaunchKernelEx(
      &config, router_gemm_kernel<T, kBlockSize, VPT, kNumTokens, kNumExperts, kHiddenDim>, output,
      mat_a, mat_b);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaLaunchKernelEx failed with error code " << cudaGetErrorString(status);
}

template void invokeRouterGemm<__nv_bfloat16, 1, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 2, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 3, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 4, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 5, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 6, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 7, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 8, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 9, 256, 7168>(float*, __nv_bfloat16 const*,
                                                            __nv_bfloat16 const*, cudaStream_t,
                                                            bool);

template void invokeRouterGemm<__nv_bfloat16, 10, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 11, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 12, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 13, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 14, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 15, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template void invokeRouterGemm<__nv_bfloat16, 16, 256, 7168>(float*, __nv_bfloat16 const*,
                                                             __nv_bfloat16 const*, cudaStream_t,
                                                             bool);

template <int kBegin, int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller {
  static void unroll(int num_tokens, float* output, __nv_bfloat16 const* input,
                     __nv_bfloat16 const* weights, cudaStream_t stream, bool launch_with_pdl) {
    if (num_tokens == kBegin) {
      invokeRouterGemm<__nv_bfloat16, kBegin, kNumExperts, kHiddenDim>(output, input, weights,
                                                                       stream, launch_with_pdl);
    } else {
      LoopUnroller<kBegin + 1, kEnd, kNumExperts, kHiddenDim>::unroll(
          num_tokens, output, input, weights, stream, launch_with_pdl);
    }
  }
};

template <int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller<kEnd, kEnd, kNumExperts, kHiddenDim> {
  static void unroll(int num_tokens, float* output, __nv_bfloat16 const* input,
                     __nv_bfloat16 const* weights, cudaStream_t stream, bool launch_with_pdl) {
    if (num_tokens == kEnd) {
      invokeRouterGemm<__nv_bfloat16, kEnd, kNumExperts, kHiddenDim>(output, input, weights, stream,
                                                                     launch_with_pdl);
    } else {
      throw std::invalid_argument("Invalid num_tokens, only supports 1 to 16");
    }
  }
};

void dsv3_router_gemm_op(TensorView mat_a, TensorView mat_b, TensorView out, bool launch_with_pdl) {
  int const num_tokens = mat_a.sizes()[0];
  int const num_experts = mat_b.sizes()[1];
  int const hidden_dim = mat_a.sizes()[1];
  auto const out_dtype_ = out.dtype();
  auto const data_type = mat_a.dtype();
  constexpr int kNumExperts = 256;
  constexpr int kHiddenDim = 7168;
  std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[1]};
  TVM_FFI_ICHECK(mat_a.dim() == 2 && mat_b.dim() == 2) << "mat_a and mat_b must be 2D tensors";
  TVM_FFI_ICHECK(mat_a.strides()[1] == 1 && out.strides()[1] == 1)
      << "mat_a and out must be row-major";
  TVM_FFI_ICHECK(mat_b.strides()[0] == 1) << "mat_b must be column-major";
  auto stream = get_stream(mat_a.device());
  bool use_custom_kernel = false;
  if (num_tokens >= 1 && num_tokens <= 16 && num_experts == kNumExperts &&
      hidden_dim == kHiddenDim && encode_dlpack_dtype(data_type) == bfloat16_code &&
      encode_dlpack_dtype(out_dtype_) == float32_code) {
    use_custom_kernel = true;
  }

  if (use_custom_kernel) {
    LoopUnroller<1, 16, kNumExperts, kHiddenDim>::unroll(
        num_tokens, reinterpret_cast<float*>(out.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream, launch_with_pdl);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input tensor size";
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv3_router_gemm_op,
                              flashinfer::trtllm_dsv3_router_gemm::dsv3_router_gemm_op);

}  // namespace flashinfer::trtllm_dsv3_router_gemm
