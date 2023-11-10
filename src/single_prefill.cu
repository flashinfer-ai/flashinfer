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
#include <thrust/device_vector.h>

#include <flashinfer/prefill.cuh>

using flashinfer::QKVLayout;
using flashinfer::RotaryMode;

template <typename dtype_in, typename dtype_out>
void bench_flashinfer_single_prefill() {
  size_t kv_len = 129;
  size_t qo_len = 129;
  size_t num_qo_heads = 32;
  size_t num_kv_heads = 32;
  size_t head_dim = 128;
  size_t rotary_mode = 0U;
  size_t layout = 0U;
  bool causal = 0U;
  bool cooperative = 0U;
  // Allocate input data:
  thrust::device_vector<dtype_in> Q(qo_len * num_qo_heads * head_dim);
  thrust::device_vector<dtype_in> K(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<dtype_in> V(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<dtype_out> O(qo_len * num_qo_heads * head_dim);
  thrust::device_vector<float> tmp(4 * 1024 * 1024);

  // Provide throughput information:
    cudaError_t status = flashinfer::SinglePrefillWithKVCache<dtype_in, dtype_out>(
        thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
        thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()),
        cooperative ? thrust::raw_pointer_cast(tmp.data()) : nullptr, num_qo_heads, num_kv_heads,
        qo_len, kv_len, head_dim, causal, QKVLayout(layout), RotaryMode(rotary_mode), 1.f, 1e4,
        nullptr);
}

int main() {
  bench_flashinfer_single_prefill<half, half>();
}