#include "flashinfer/gemm/dsv3_router_gemm.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::trtllm_dsv3_router_gemm {

// Note: Explicit template instantiations are not needed here because
// LoopUnroller already forces instantiation of all required specializations.
template <typename Tin, typename Tout, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemm(Tout* output, Tin const* mat_a, Tin const* mat_b, cudaStream_t stream,
                      bool use_pdl = false) {
  constexpr int VPT = 16 / sizeof(Tin);
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
      &config, router_gemm_kernel<Tin, Tout, kBlockSize, VPT, kNumTokens, kNumExperts, kHiddenDim>,
      output, mat_a, mat_b);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaLaunchKernelEx failed with error code " << cudaGetErrorString(status);
}

template <int kBegin, int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller {
  template <typename Tout>
  static void unroll(int num_tokens, Tout* output, __nv_bfloat16 const* input,
                     __nv_bfloat16 const* weights, cudaStream_t stream, bool launch_with_pdl) {
    if (num_tokens == kBegin) {
      invokeRouterGemm<__nv_bfloat16, Tout, kBegin, kNumExperts, kHiddenDim>(
          output, input, weights, stream, launch_with_pdl);
    } else {
      LoopUnroller<kBegin + 1, kEnd, kNumExperts, kHiddenDim>::unroll(
          num_tokens, output, input, weights, stream, launch_with_pdl);
    }
  }
};

template <int kEnd, int kNumExperts, int kHiddenDim>
struct LoopUnroller<kEnd, kEnd, kNumExperts, kHiddenDim> {
  template <typename Tout>
  static void unroll(int num_tokens, Tout* output, __nv_bfloat16 const* input,
                     __nv_bfloat16 const* weights, cudaStream_t stream, bool launch_with_pdl) {
    if (num_tokens == kEnd) {
      invokeRouterGemm<__nv_bfloat16, Tout, kEnd, kNumExperts, kHiddenDim>(output, input, weights,
                                                                           stream, launch_with_pdl);
    } else {
      throw std::invalid_argument("Invalid num_tokens, only supports 1 to 16");
    }
  }
};

template <typename Tout, int64_t tout_code, int kNumExperts, int kBegin, int kEnd>
void generic_router_gemm_op(TensorView mat_a, TensorView mat_b, TensorView out,
                            bool launch_with_pdl) {
  int const num_tokens = mat_a.sizes()[0];
  int const num_experts = mat_b.sizes()[1];
  int const hidden_dim = mat_a.sizes()[1];
  auto const out_dtype_ = out.dtype();
  auto const data_type = mat_a.dtype();
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
      encode_dlpack_dtype(out_dtype_) == tout_code) {
    use_custom_kernel = true;
  }

  if (use_custom_kernel) {
    LoopUnroller<kBegin, kEnd, kNumExperts, kHiddenDim>::unroll(
        num_tokens, reinterpret_cast<Tout*>(out.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(mat_a.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(mat_b.data_ptr()), stream, launch_with_pdl);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input tensor size";
  }
}

void dsv3_router_gemm_op(TensorView mat_a, TensorView mat_b, TensorView out, bool launch_with_pdl) {
  generic_router_gemm_op<float, float32_code, 256, 1, 16>(mat_a, mat_b, out, launch_with_pdl);
}

void ml3_router_gemm_op(TensorView mat_a, TensorView mat_b, TensorView out, bool launch_with_pdl) {
  generic_router_gemm_op<__nv_bfloat16, bfloat16_code, 128, 1, 16>(mat_a, mat_b, out,
                                                                   launch_with_pdl);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv3_router_gemm_op,
                              flashinfer::trtllm_dsv3_router_gemm::dsv3_router_gemm_op);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ml3_router_gemm_op,
                              flashinfer::trtllm_dsv3_router_gemm::ml3_router_gemm_op);

}  // namespace flashinfer::trtllm_dsv3_router_gemm
