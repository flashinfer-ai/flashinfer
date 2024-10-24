/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 *Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch
// headers.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/numeric_types.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#include <flashinfer/attention/hopper/prefill_sm90.cuh>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

using namespace flashinfer;

std::vector<at::Tensor> single_prefill_with_kv_cache(
    at::Tensor& q,                    // qo_len x num_qo_heads x head_dim
    at::Tensor& k,                    // kv_len x num_kv_heads x head_dim
    at::Tensor& v,                    // kv_len x num_kv_heads x head_dim
    at::Tensor& out_,  // qo_len x num_qo_heads x head_dim
    bool causal,                      // whether to apply causal masking
    float sm_scale) {
  auto q_dtype = q.dtype();

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};
  int64_t num_qo_heads = q.size(1);
  int64_t qo_len = q.size(0);

  auto opts = q.options();
  auto lse = torch::empty({num_qo_heads, qo_len}, opts.dtype(at::kFloat));
  SinglePrefillParams<cutlass::half_t, cutlass::half_t, cutlass::half_t> params; 
  params.q_ptr = static_cast<cutlass::half_t*>(q.data_ptr());
  params.k_ptr = static_cast<cutlass::half_t*>(k.data_ptr());
  params.v_ptr = static_cast<cutlass::half_t*>(v.data_ptr());
  params.o_ptr = static_cast<cutlass::half_t*>(out_.data_ptr());
  params.lse_ptr = static_cast<float*>(lse.data_ptr());
  params.q_stride_n = q.stride(0);
  params.k_stride_n = k.stride(0);
  params.v_stride_n = v.stride(0);
  params.o_stride_n = out_.stride(0);
  params.q_stride_h = q.stride(1);
  params.k_stride_h = k.stride(1);
  params.v_stride_h = v.stride(1);
  params.o_stride_h = out_.stride(1);
  params.qo_len = q.size(0);
  params.kv_len = k.size(0);
  params.head_dim = q.size(2);
  params.num_qo_heads = q.size(1);
  params.num_kv_heads = k.size(1);
  params.group_size = params.num_qo_heads / params.num_kv_heads;
  params.sm_scale_log2 = sm_scale * 1.44269504088896340736f;

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  SinglePrefillWithKVCache(params, stream);

  return {out_, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("single_prefill_with_kv_cache", &single_prefill_with_kv_cache);
}
