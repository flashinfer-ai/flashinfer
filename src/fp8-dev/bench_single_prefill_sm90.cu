/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/numeric_types.h>
#include <thrust/device_vector.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#include <flashinfer/attention/hopper/default_params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/quantization/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <nvbench/nvbench.cuh>
#include <optional>

#include "flashattention_ops.h"
#include "utils.h"

namespace flashinfer {

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLINDING_WINDOW,
          typename AttentionVariant, typename Params>
cudaError_t SingleFP8PrefillWithKVCacheDispatched(Params& params, cudaStream_t stream);

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLINDING_WINDOW,
          typename AttentionVariant, typename Params>
cudaError_t SinglePrefillWithKVCacheDispatched(Params& params, cudaStream_t stream);
}  // namespace flashinfer

using namespace flashinfer;

void single_fp8_prefill_with_kv_cache_sm90(nvbench::state& state) {
  size_t qo_len = state.get_int64("seq_len");
  size_t kv_len = state.get_int64("seq_len");
  size_t num_qo_heads = state.get_int64("num_qo_heads");
  size_t num_kv_heads = state.get_int64("num_kv_heads");
  size_t head_dim = state.get_int64("head_dim");
  QKVLayout kv_layout = QKVLayout(state.get_int64("kv_layout"));
  MaskMode mask_mode = MaskMode(state.get_int64("mask_mode"));

  if (qo_len > kv_len) {
    state.skip("qo_len should be less than kv_len");
  }

  using DTypeQ = cutlass::float_e4m3_t;
  using DTypeKV = cutlass::float_e4m3_t;
  using DTypeO = cutlass::half_t;
  using IdType = int32_t;

  constexpr auto USE_SLIDING_WINDOW = false;

  using Params = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO, IdType>;
  using AttentionVariant = DefaultFP8Attention;

  thrust::device_vector<DTypeQ> q(qo_len * num_qo_heads * head_dim);
  thrust::device_vector<DTypeKV> k(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<DTypeKV> v(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<DTypeO> o(qo_len * num_qo_heads * head_dim);

  thrust::device_vector<float> scale_q(num_qo_heads);
  thrust::device_vector<float> scale_k(num_kv_heads);
  thrust::device_vector<float> scale_v(num_kv_heads);

  Params params;
  params.q_ptr = static_cast<DTypeQ*>(thrust::raw_pointer_cast(q.data()));
  params.k_ptr = static_cast<DTypeKV*>(thrust::raw_pointer_cast(k.data()));
  params.v_ptr = static_cast<DTypeKV*>(thrust::raw_pointer_cast(v.data()));
  params.o_ptr = static_cast<DTypeO*>(thrust::raw_pointer_cast(o.data()));
  params.lse_ptr = nullptr;
  // q NHD
  params.q_stride_n = num_qo_heads * head_dim;
  params.q_stride_h = head_dim;
  params.o_stride_n = num_qo_heads * head_dim;
  params.o_stride_h = head_dim;
  if (kv_layout == QKVLayout::kNHD) {
    params.k_stride_n = num_kv_heads * head_dim;
    params.k_stride_h = head_dim;
    params.v_stride_n = num_kv_heads * head_dim;
    params.v_stride_h = head_dim;
  } else {
    // k HND
    params.k_stride_h = kv_len * head_dim;
    params.k_stride_n = head_dim;
    params.v_stride_h = kv_len * head_dim;
    params.v_stride_n = head_dim;
  }
  params.qo_len = qo_len;
  params.kv_len = kv_len;
  params.head_dim = head_dim;
  params.num_qo_heads = num_qo_heads;
  params.num_kv_heads = num_kv_heads;
  params.causal = mask_mode == MaskMode::kCausal;
  params.group_size = params.num_qo_heads / params.num_kv_heads;
  params.window_left = 0;

  params.additional_params.scale_q = thrust::raw_pointer_cast(scale_q.data());
  params.additional_params.scale_k = thrust::raw_pointer_cast(scale_k.data());
  params.additional_params.scale_v = thrust::raw_pointer_cast(scale_v.data());
  params.additional_params.sm_scale = 1.f / std::sqrt(float(head_dim));

  state.add_global_memory_reads<uint8_t>(
      (qo_len * num_qo_heads + 2 * kv_len * num_kv_heads) * sizeof(DTypeQ) * head_dim, "Read");
  state.add_global_memory_writes<half>(qo_len * num_qo_heads * head_dim, "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();

    cudaError_t status;
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
        status = SingleFP8PrefillWithKVCacheDispatched<HEAD_DIM, MASK_MODE, USE_SLIDING_WINDOW,
                                                       AttentionVariant, Params>(
            params, launch.get_stream());
      });
    });
    if (status != cudaSuccess) {
      state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
    timer.stop();
    cudaDeviceSynchronize();
  });

  const auto measured_mean = static_cast<nvbench::float32_t>(
      state.get_summary("nv/cold/time/gpu/mean").get_float64("value"));
  auto& summ = state.add_summary("nv/tflops");
  summ.set_string("description", "Achieved TFlops/s");
  summ.set_string("name", "TFlops/s");
  float tflops;
  if (params.causal) {
    tflops = qo_len * (2 * kv_len - qo_len) * 2 * num_kv_heads * head_dim / measured_mean / 1e12;
  } else {
    tflops = qo_len * kv_len * 4 * num_kv_heads * head_dim / measured_mean / 1e12;
  }
  summ.set_float64("value", tflops);
}

void single_fp16_prefill_with_kv_cache_sm90(nvbench::state& state) {
  size_t qo_len = state.get_int64("seq_len");
  size_t kv_len = state.get_int64("seq_len");
  size_t num_qo_heads = state.get_int64("num_qo_heads");
  size_t num_kv_heads = state.get_int64("num_kv_heads");
  size_t head_dim = state.get_int64("head_dim");
  QKVLayout kv_layout = QKVLayout(state.get_int64("kv_layout"));
  MaskMode mask_mode = MaskMode(state.get_int64("mask_mode"));

  if (qo_len > kv_len) {
    state.skip("qo_len should be less than kv_len");
  }

  using DTypeQ = cutlass::half_t;
  using DTypeKV = cutlass::half_t;
  using DTypeO = cutlass::half_t;
  using IdType = int32_t;

  constexpr auto USE_SLIDING_WINDOW = false;

  using Params = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO, IdType>;
  using AttentionVariant = StandardAttention;

  thrust::device_vector<DTypeQ> q(qo_len * num_qo_heads * head_dim);
  thrust::device_vector<DTypeKV> k(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<DTypeKV> v(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<DTypeO> o(qo_len * num_qo_heads * head_dim);

  Params params;
  params.q_ptr = static_cast<DTypeQ*>(thrust::raw_pointer_cast(q.data()));
  params.k_ptr = static_cast<DTypeKV*>(thrust::raw_pointer_cast(k.data()));
  params.v_ptr = static_cast<DTypeKV*>(thrust::raw_pointer_cast(v.data()));
  params.o_ptr = static_cast<DTypeO*>(thrust::raw_pointer_cast(o.data()));
  params.lse_ptr = nullptr;
  // q NHD
  params.q_stride_n = num_qo_heads * head_dim;
  params.q_stride_h = head_dim;
  params.o_stride_n = num_qo_heads * head_dim;
  params.o_stride_h = head_dim;
  if (kv_layout == QKVLayout::kNHD) {
    params.k_stride_n = num_kv_heads * head_dim;
    params.k_stride_h = head_dim;
    params.v_stride_n = num_kv_heads * head_dim;
    params.v_stride_h = head_dim;
  } else {
    // k HND
    params.k_stride_h = kv_len * head_dim;
    params.k_stride_n = head_dim;
    params.v_stride_h = kv_len * head_dim;
    params.v_stride_n = head_dim;
  }
  params.qo_len = qo_len;
  params.kv_len = kv_len;
  params.head_dim = head_dim;
  params.num_qo_heads = num_qo_heads;
  params.num_kv_heads = num_kv_heads;
  params.causal = mask_mode == MaskMode::kCausal;
  params.group_size = params.num_qo_heads / params.num_kv_heads;
  params.window_left = 0;
  params.additional_params.sm_scale = 1.f / std::sqrt(float(head_dim));

  state.add_global_memory_reads<uint8_t>(
      (qo_len * num_qo_heads + 2 * kv_len * num_kv_heads) * sizeof(DTypeQ) * head_dim, "Read");
  state.add_global_memory_writes<half>(qo_len * num_qo_heads * head_dim, "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();

    cudaError_t status;
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
        status = SinglePrefillWithKVCacheDispatched<HEAD_DIM, MASK_MODE, USE_SLIDING_WINDOW,
                                                    AttentionVariant, Params>(params,
                                                                              launch.get_stream());
      });
    });

    if (status != cudaSuccess) {
      state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
    timer.stop();
  });

  const auto measured_mean = static_cast<nvbench::float32_t>(
      state.get_summary("nv/cold/time/gpu/mean").get_float64("value"));
  auto& summ = state.add_summary("nv/tflops");
  summ.set_string("description", "Achieved TFlops/s");
  summ.set_string("name", "TFlops/s");
  float tflops;
  if (params.causal) {
    tflops = qo_len * (2 * kv_len - qo_len) * 2 * num_kv_heads * head_dim / measured_mean / 1e12;
  } else {
    tflops = qo_len * kv_len * 4 * num_kv_heads * head_dim / measured_mean / 1e12;
  }
  summ.set_float64("value", tflops);
}

void single_fp8_fa3_prefill_with_kv_cache_sm90(nvbench::state& state) {
  size_t qo_len = state.get_int64("seq_len");
  size_t kv_len = state.get_int64("seq_len");
  size_t num_qo_heads = state.get_int64("num_qo_heads");
  size_t num_kv_heads = state.get_int64("num_kv_heads");
  size_t head_dim = state.get_int64("head_dim");
  MaskMode mask_mode = MaskMode(state.get_int64("mask_mode"));

  bool is_causal = mask_mode == MaskMode::kCausal;
  float sm_scale = 1.f / std::sqrt(float(head_dim));

  if (qo_len > kv_len) {
    state.skip("qo_len should be less than kv_len");
  }

  using DTypeQ = cutlass::float_e4m3_t;
  using DTypeKV = cutlass::float_e4m3_t;
  using DTypeO = cutlass::half_t;
  using IdType = int32_t;
  using DTypeScale = float;

  std::vector<DTypeQ> q(qo_len * num_qo_heads * head_dim);
  std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim);
  std::vector<DTypeKV> v(kv_len * num_kv_heads * head_dim);
  std::vector<DTypeO> o(qo_len * num_qo_heads * head_dim);
  std::vector<DTypeScale> scale_q(1);
  std::vector<DTypeScale> scale_k(1);
  std::vector<DTypeScale> scale_v(1);

  utils::vec_normal_(q);
  utils::vec_normal_(k);
  utils::vec_normal_(v);

  utils::vec_zero_(o);
  utils::vec_normal_(scale_q);
  utils::vec_normal_(scale_k);
  utils::vec_normal_(scale_v);

  auto device = torch::Device(torch::kCUDA, 0);
  auto q_t =
      torch::from_blob(q.data(), {1, uint32_t(qo_len), uint32_t(num_qo_heads), uint32_t(head_dim)},
                       torch::kFloat8_e4m3fn)
          .clone()
          .to(device);
  auto k_t =
      torch::from_blob(k.data(), {1, uint32_t(kv_len), uint32_t(num_kv_heads), uint32_t(head_dim)},
                       torch::kFloat8_e4m3fn)
          .clone()
          .to(device);
  auto v_t =
      torch::from_blob(v.data(), {1, uint32_t(kv_len), uint32_t(num_kv_heads), uint32_t(head_dim)},
                       torch::kFloat8_e4m3fn)
          .clone()
          .to(device);
  auto scale_q_t = std::optional<at::Tensor>(
      torch::from_blob(scale_q.data(), {1}, torch::kFloat).clone().to(device));
  auto scale_k_t = std::optional<at::Tensor>(
      torch::from_blob(scale_k.data(), {1}, torch::kFloat).clone().to(device));
  auto scale_v_t = std::optional<at::Tensor>(
      torch::from_blob(scale_v.data(), {1}, torch::kFloat).clone().to(device));
  auto out_t = std::optional<at::Tensor>{};

  state.add_global_memory_reads<uint8_t>(
      (qo_len * num_qo_heads + 2 * kv_len * num_kv_heads) * sizeof(DTypeQ) * head_dim, "Read");
  state.add_global_memory_writes<half>(qo_len * num_qo_heads * head_dim, "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();

    auto o_t_ref =
        mha_fwd(q_t, k_t, v_t, out_t, sm_scale, scale_q_t, scale_k_t, scale_v_t, is_causal)[0];

    timer.stop();
    cudaDeviceSynchronize();
  });

  const auto measured_mean = static_cast<nvbench::float32_t>(
      state.get_summary("nv/cold/time/gpu/mean").get_float64("value"));
  auto& summ = state.add_summary("nv/tflops");
  summ.set_string("description", "Achieved TFlops/s");
  summ.set_string("name", "TFlops/s");
  float tflops;
  if (is_causal) {
    tflops = qo_len * (2 * kv_len - qo_len) * 2 * num_kv_heads * head_dim / measured_mean / 1e12;
  } else {
    tflops = qo_len * kv_len * 4 * num_kv_heads * head_dim / measured_mean / 1e12;
  }
  summ.set_float64("value", tflops);
}

NVBENCH_BENCH(single_fp8_prefill_with_kv_cache_sm90)
    .set_name(("single_fp8_prefill_with_kv_cache_sm90"))
    .add_int64_axis("seq_len", {2048, 4096, 8192, 16384})
    .add_int64_axis("num_qo_heads", {32})
    .add_int64_axis("num_kv_heads", {32})
    .add_int64_axis("head_dim", {64, 128, 256})
    .add_int64_axis("mask_mode", {0, 1})
    .add_int64_axis("kv_layout", {0});

NVBENCH_BENCH(single_fp16_prefill_with_kv_cache_sm90)
    .set_name(("single_fp16_prefill_with_kv_cache_sm90"))
    .add_int64_axis("seq_len", {2048, 4096, 8192, 16384})
    .add_int64_axis("num_qo_heads", {32})
    .add_int64_axis("num_kv_heads", {32})
    .add_int64_axis("head_dim", {64, 128, 256})
    .add_int64_axis("mask_mode", {0, 1})
    .add_int64_axis("kv_layout", {0});

NVBENCH_BENCH(single_fp8_fa3_prefill_with_kv_cache_sm90)
    .set_name(("single_fp8_fa3_prefill_with_kv_cache_sm90"))
    .add_int64_axis("seq_len", {2048, 4096, 8192, 16384})
    .add_int64_axis("num_qo_heads", {32})
    .add_int64_axis("num_kv_heads", {32})
    .add_int64_axis("head_dim", {64, 128, 256})
    .add_int64_axis("mask_mode", {0, 1});