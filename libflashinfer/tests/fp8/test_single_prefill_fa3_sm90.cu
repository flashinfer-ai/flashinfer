/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include <torch/nn/functional.h>
#include <torch/python.h>

#include <cstdint>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/layout.cuh>

#include "flashinfer/utils/fp8/cpu_reference.h"
#include "flashinfer/utils/fp8/flashattention_ops.h"
#include "flashinfer/utils/fp8/utils.h"
using namespace flashinfer;

void run_fwd(thrust::device_vector<cutlass::float_e4m3_t>& q_d,
             thrust::device_vector<cutlass::float_e4m3_t>& k_d,
             thrust::device_vector<cutlass::float_e4m3_t>& v_d,
             thrust::device_vector<cutlass::half_t>& o_d, thrust::device_vector<float>& scale_q_d,
             thrust::device_vector<float>& scale_k_d, thrust::device_vector<float>& scale_v_d,
             int32_t qo_len, int32_t kv_len, int32_t num_qo_heads, int32_t num_kv_heads,
             int32_t head_dim, float sm_scale, MaskMode mask_mode, QKVLayout kv_layout);

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
void _TestSingleFP8PrefillKernelCorrectness(int32_t qo_len, int32_t kv_len, int32_t num_qo_heads,
                                            int32_t num_kv_heads, int32_t head_dim,
                                            MaskMode mask_mode, QKVLayout kv_layout, size_t seed,
                                            float rtol = 1e-3, float atol = 1e-3) {
  using DTypeScale = float;
  float sm_scale = 1.f / std::sqrt(float(head_dim));

  std::vector<DTypeQ> q(qo_len * num_qo_heads * head_dim);
  std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim);
  std::vector<DTypeKV> v(kv_len * num_kv_heads * head_dim);
  std::vector<DTypeO> o(qo_len * num_qo_heads * head_dim);
  std::vector<DTypeScale> scale_q(num_qo_heads);
  std::vector<DTypeScale> scale_k(num_kv_heads);
  std::vector<DTypeScale> scale_v(num_kv_heads);

  std::vector<float> q_fp32(qo_len * num_qo_heads * head_dim);
  std::vector<float> k_fp32(kv_len * num_kv_heads * head_dim);
  std::vector<float> v_fp32(kv_len * num_kv_heads * head_dim);

  utils::vec_normal_(q_fp32, 0, 16, seed);
  utils::vec_normal_(k_fp32, 0, 16, seed);
  utils::vec_normal_(v_fp32, 0, 16, seed);
  utils::vec_zero_(o);

  cpu_reference::sym_quant_per_head(q_fp32, q, scale_q, qo_len, num_qo_heads, head_dim, kv_layout,
                                    true);
  cpu_reference::sym_quant_per_head(k_fp32, k, scale_k, kv_len, num_kv_heads, head_dim, kv_layout,
                                    false);
  cpu_reference::sym_quant_per_head(v_fp32, v, scale_v, kv_len, num_kv_heads, head_dim, kv_layout,
                                    false);

  thrust::device_vector<DTypeQ> q_d(q);
  thrust::device_vector<DTypeKV> k_d(k);
  thrust::device_vector<DTypeKV> v_d(v);
  thrust::device_vector<DTypeO> o_d(o);
  thrust::device_vector<DTypeScale> scale_q_d(scale_q);
  thrust::device_vector<DTypeScale> scale_k_d(scale_k);
  thrust::device_vector<DTypeScale> scale_v_d(scale_v);

  run_fwd(q_d, k_d, v_d, o_d, scale_q_d, scale_k_d, scale_v_d, qo_len, kv_len, num_qo_heads,
          num_kv_heads, head_dim, sm_scale, mask_mode, kv_layout);

  thrust::host_vector<DTypeO> o_flashinfer_copy(o_d);
  std::vector<DTypeO> o_flashinfer(o_flashinfer_copy.begin(), o_flashinfer_copy.end());

  /*
      Below is FA3 implementation API call
  */
  auto device = torch::Device(torch::kCUDA, 0);
  auto q_t = torch::from_blob(q.data(), {1, qo_len, num_qo_heads, head_dim}, torch::kFloat8_e4m3fn)
                 .clone()
                 .to(device);
  auto k_t = torch::from_blob(k.data(), {1, kv_len, num_kv_heads, head_dim}, torch::kFloat8_e4m3fn)
                 .clone()
                 .to(device);
  auto v_t = torch::from_blob(v.data(), {1, kv_len, num_kv_heads, head_dim}, torch::kFloat8_e4m3fn)
                 .clone()
                 .to(device);
  auto scale_q_t = std::optional<at::Tensor>(
      torch::from_blob(scale_q.data(), {1}, torch::kFloat).clone().to(device));
  auto scale_k_t = std::optional<at::Tensor>(
      torch::from_blob(scale_k.data(), {1}, torch::kFloat).clone().to(device));
  auto scale_v_t = std::optional<at::Tensor>(
      torch::from_blob(scale_v.data(), {1}, torch::kFloat).clone().to(device));
  auto out_t = std::optional<at::Tensor>{};

  bool is_causal = mask_mode == MaskMode::kCausal;
  auto o_t_ref =
      mha_fwd(q_t, k_t, v_t, out_t, sm_scale, scale_q_t, scale_k_t, scale_v_t, is_causal)[0].to(
          torch::Device(torch::kCPU));
  DTypeO* o_ref_ptr = static_cast<DTypeO*>(o_t_ref.data_ptr());
  std::vector<DTypeO> o_fa3(o_ref_ptr, o_ref_ptr + o_t_ref.numel());

  std::vector<DTypeO> o_cpu = cpu_reference::single_fp8_mha<DTypeQ, DTypeO, DTypeScale>(
      q, k, v, scale_q, scale_k, scale_v, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim,
      sm_scale, is_causal, kv_layout);

  std::vector<DTypeO> o_gold = cpu_reference::single_mha<float, DTypeO>(
      q_fp32, k_fp32, v_fp32, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, sm_scale,
      is_causal, kv_layout);

  float fa3_mse = utils::vec_cal_mse_(o_fa3, o_gold);
  float finfer_mse = utils::vec_cal_mse_(o_flashinfer, o_gold);
  float fcpu_mse = utils::vec_cal_mse_(o_cpu, o_gold);
  printf("%d,%d,%d,%d,%d,%d,%f,%f,%f\n", num_qo_heads, num_kv_heads, qo_len, kv_len, head_dim,
         int(mask_mode), fcpu_mse, fa3_mse, finfer_mse);
}

template <typename DTypeIn, typename DTypeO>
void TestSingleFP8PrefillKernelLongContextCorrectness() {
  for (size_t seq_len : {128, 256, 512, 1024, 2048, 4096}) {
    for (size_t num_qo_heads : {1}) {
      for (size_t num_kv_heads : {1}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (size_t kv_layout : {0}) {
            for (size_t mask_mode : {0, 1}) {
              for (size_t seed : {600})
                _TestSingleFP8PrefillKernelCorrectness<DTypeIn, DTypeIn, DTypeO>(
                    seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, MaskMode(mask_mode),
                    QKVLayout(kv_layout), seed);
            }
          }
        }
      }
    }
  }
}

int main() {
  printf(
      "num_qo_heads,num_kv_heads,qo_len,kv_len,head_dim,mask_mode,cpu_mse,fa3_mse,"
      "flashinfer_mse\n");
  TestSingleFP8PrefillKernelLongContextCorrectness<cutlass::float_e4m3_t, cutlass::half_t>();
  return 0;
}
