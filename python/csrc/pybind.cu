#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "../../include/flashinfer.cuh"

#define DISPATCH_SWITCH(torch_dtype, dtype, ...) \
  switch (torch_dtype) { \
    case torch::ScalarType::Float: \
      using dtype = float; \
      __VA_ARGS__ \
      break; \
    case torch::ScalarType::Half: \
      using dtype = half; \
      __VA_ARGS__ \
      break; \
    case torch::ScalarType::BFloat16: \
      using dtype = nv_bfloat16; \
      __VA_ARGS__ \
    default: \
      AT_ERROR("Unsupported dtype"); \
  }

torch::Tensor SingleDecodeWithKVCache(
  torch::Tensor Q,
  torch::Tensor K,
  torch::Tensor V,
  torch::ScalarType dtype
) {
  assert(Q.is_contiguous());
  assert(K.is_contiguous());
  assert(V.is_contiguous());
  assert(Q.scalar_type() == K.scalar_type());
  assert(Q.scalar_type() == V.scalar_type());
  assert(Q.dim() == 2);
  assert(K.dim() == 3);
  assert(V.dim() == 3);
  int num_heads = Q.size(0),
      head_dim = Q.size(1);
  int seq_len = K.size(0);
  assert(K.size(1) == num_heads);
  assert(K.size(2) == head_dim);
  assert(V.size(0) == seq_len);
  assert(V.size(1) == num_heads);
  assert(V.size(2) == head_dim);
  auto output_options = Q.options().dtype(dtype);
  torch::Tensor output = torch::empty({num_heads, head_dim}, output_options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_SWITCH(Q.scalar_dtype(), dtype_in, {
    DISPATCH_SWITCH(dtype, dtype_out, {

    });
  });

  // AT_DISPATCH_FLOATING_TYPES()
  // flashinfer::single_decode_with_kv_cache(
  //   Q.data_ptr<float>(),
  //   K.data_ptr<float>(),
  //   V.data_ptr<float>(),
  //   output.data_ptr<float>(),

  // );
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_single_decode", &SingleDecode, "CUDA implementation of decoding-only self-attention for batch_size=1.");
}