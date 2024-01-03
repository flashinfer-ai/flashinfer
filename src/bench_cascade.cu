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

#include <flashinfer/cascade.cuh>
#include <nvbench/nvbench.cuh>

#include "utils.h"

template <typename T>
void bench_merge_states(nvbench::state& state) {
  const auto num_index_sets = state.get_int64("num_index_sets");
  const auto batch_size = state.get_int64("batch_size");
  const auto num_heads = state.get_int64("num_heads");
  const auto head_dim = state.get_int64("head_dim");

  std::vector<T> V_host(num_index_sets * batch_size * num_heads * head_dim);
  std::vector<float> S_host(num_index_sets * batch_size * num_heads);

  utils::vec_normal_(V_host);
  utils::vec_uniform_(S_host, 5, 10);

  thrust::device_vector<T> V_device(V_host);
  thrust::device_vector<float> S_device(S_host);
  thrust::device_vector<T> V_merged(batch_size * num_heads * head_dim);
  thrust::device_vector<float> S_merged(batch_size * num_heads);

  state.add_global_memory_reads<T>(V_host.size(), "Read");
  state.add_global_memory_writes<T>(V_merged.size(), "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status = flashinfer::MergeStates(
        thrust::raw_pointer_cast(V_device.data()), thrust::raw_pointer_cast(S_device.data()),
        thrust::raw_pointer_cast(V_merged.data()), thrust::raw_pointer_cast(S_merged.data()),
        num_index_sets, batch_size, num_heads, head_dim);
    timer.stop();
  });
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_MERGE_KERNELS(T)                               \
  auto bench_flashinfer_merge_states_##T##_ = bench_merge_states<T>;    \
  NVBENCH_BENCH(bench_flashinfer_merge_states_##T##_)                   \
      .set_name("flashinfer_merge_states_" STR(T))                      \
      .add_int64_axis("num_index_sets", {2, 16, 64, 128, 256})          \
      .add_int64_axis("batch_size", {1, 2, 4, 8, 16, 32, 64, 128, 256}) \
      .add_int64_axis("num_heads", {32})                                \
      .add_int64_axis("head_dim", {128})

BENCH_FLASHINFER_MERGE_KERNELS(half);
