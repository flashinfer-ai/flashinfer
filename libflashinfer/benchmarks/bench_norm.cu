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
#include <thrust/device_vector.h>

#include <flashinfer/norm.cuh>
#include <nvbench/nvbench.cuh>

#include "utils.h"

using namespace flashinfer;

template <typename T>
void bench_rms_norm(nvbench::state& state) {
  size_t batch_size = state.get_int64("batch_size");
  size_t hidden_dim = state.get_int64("hidden_dim");

  thrust::device_vector<T> x(batch_size * hidden_dim);
  thrust::device_vector<T> w(hidden_dim);
  thrust::device_vector<T> y(batch_size * hidden_dim);

  state.add_global_memory_reads<T>(batch_size * hidden_dim + hidden_dim, "Read");
  state.add_global_memory_writes<T>(batch_size * hidden_dim, "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
    timer.start();
    cudaError_t status =
        norm::RMSNorm<T>(thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(w.data()),
                         thrust::raw_pointer_cast(y.data()), batch_size, hidden_dim, 1e-5);
    timer.stop();
    if (status != cudaSuccess) {
      state.skip("RMSNorm kernel launch failed");
    }
  });
}

auto bench_rms_norm_f16 = bench_rms_norm<half>;
NVBENCH_BENCH(bench_rms_norm_f16)
    .set_name("bench_rms_norm_f16")
    .add_int64_axis("batch_size", {32, 128, 512, 2048})
    .add_int64_axis("hidden_dim", {3072, 4096, 32768});
