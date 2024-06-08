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
#include <driver_types.h>
#include <mpi.h>
#include <spdlog/spdlog.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/host_vector.h>

#include <cstdint>
#include <flashinfer/attention/cascade.cuh>
#include <flashinfer/distributed/all_reduce.cuh>
#include <flashinfer/utils.cuh>

#include "utils.h"

using namespace flashinfer;

int main(int argc, char* argv[]) {
  // init mpi
  MPI_Init(&argc, &argv);
  spdlog::info("MPI Initialized.");
  int nranks, rank;
  // get work size and rank id
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaSetDevice(rank);
  spdlog::info("nranks: {}, rank: {}", nranks, rank);
  // init host and device buffers
  using T = half;
  const uint32_t batch_size = 64;
  const uint32_t num_heads = 64;
  const uint32_t head_dim = 128;
  const uint32_t aligned_buf_size_in_bytes =
      ceil_div(
          (batch_size * num_heads * head_dim) * sizeof(T) + batch_size * num_heads * sizeof(float),
          128) *
      128;
  std::vector<T> V(nranks * batch_size * num_heads * head_dim),
      V_T(batch_size * nranks * num_heads * head_dim);
  std::vector<float> S(nranks * batch_size * num_heads), S_T(batch_size * nranks * num_heads);
  for (int32_t r = 0; r < nranks; ++r) {
    for (int32_t i = 0; i < batch_size; ++i) {
      for (int32_t h = 0; h < num_heads; ++h) {
        for (int32_t j = 0; j < head_dim; ++j) {
          float v = T((r + i + h + j) % 10);
          V[((r * batch_size + i) * num_heads + h) * head_dim + j] = v;
          V_T[((i * nranks + r) * num_heads + h) * head_dim + j] = v;
        }
        float s = (r + i + h) % 5 - 2;
        S[(r * batch_size + i) * num_heads + h] = s;
        S_T[(i * nranks + r) * num_heads + h] = s;
      }
    }
  }

  thrust::device_vector<T> V_T_d(V_T);
  thrust::device_vector<float> S_T_d(S_T);
  thrust::device_vector<T> V_reduce_ref(batch_size * num_heads * head_dim);
  thrust::device_vector<float> S_reduce_ref(batch_size * num_heads);

  auto status = MergeStates(
      thrust::raw_pointer_cast(V_T_d.data()), thrust::raw_pointer_cast(S_T_d.data()),
      thrust::raw_pointer_cast(V_reduce_ref.data()), thrust::raw_pointer_cast(S_reduce_ref.data()),
      nranks, batch_size, num_heads, head_dim);

  if (status != cudaSuccess) {
    spdlog::error("rank: {}, reference attention reduce kernel failed.", rank);
  }
  thrust::host_vector<T> V_reduce_ref_h = V_reduce_ref;
  thrust::host_vector<float> S_reduce_ref_h = S_reduce_ref;

  std::vector<uint8_t> host_buf(aligned_buf_size_in_bytes);
  uint8_t* host_buf_ptr = host_buf.data();
  memcpy(host_buf_ptr, V.data() + rank * batch_size * num_heads * head_dim,
         batch_size * num_heads * head_dim * sizeof(T));
  host_buf_ptr += batch_size * num_heads * head_dim * sizeof(T);
  memcpy(host_buf_ptr, S.data() + rank * batch_size * num_heads,
         batch_size * num_heads * sizeof(float));
  thrust::device_vector<uint8_t> device_buf(host_buf);

  // Initialize communicator
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, nranks);
  mscclpp::UniqueId unique_id;
  if (rank == 0) unique_id = bootstrap->createUniqueId();
  MPI_Bcast(&unique_id, sizeof(unique_id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(unique_id);
  mscclpp::Communicator comm(bootstrap);
  spdlog::info("rank: {}, communicator initialized.", rank);

  // setup sm channels
  std::vector<mscclpp::SmChannel> sm_channels;
  distributed::SetupChannels(&comm, sm_channels, rank, nranks,
                             thrust::raw_pointer_cast(device_buf.data()),
                             aligned_buf_size_in_bytes);
  std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> sm_channel_handlers(sm_channels.size());
  std::transform(
      sm_channels.begin(), sm_channels.end(), sm_channel_handlers.begin(),
      [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
  thrust::device_vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> sm_channel_handlers_d(
      sm_channel_handlers);
  spdlog::info("rank: {}, sm channels setup.", rank);

  MPI_Barrier(MPI_COMM_WORLD);

  // call attention all reduce kernel
  constexpr uint32_t vec_size = 16 / sizeof(T);
  dim3 nthrs(head_dim / vec_size / nranks, num_heads);
  dim3 nblks(batch_size);
  distributed::AttentionAllReduceInplaceKernel<T><<<nblks, nthrs>>>(
      thrust::raw_pointer_cast(sm_channel_handlers_d.data()),
      thrust::raw_pointer_cast(device_buf.data()), rank, nranks, batch_size, num_heads, head_dim);

  status = cudaGetLastError();
  if (status != cudaSuccess) {
    spdlog::error("rank: {}, cuda error: {}", rank, cudaGetErrorString(status));
  }

  // check results
  thrust::host_vector<uint8_t> host_buf_result(device_buf);
  uint8_t* host_buf_result_ptr = host_buf_result.data();
  std::vector<T> V_allreduce_h(batch_size * num_heads * head_dim);
  std::vector<float> S_allreduce_h(batch_size * num_heads);

  memcpy(V_allreduce_h.data(), host_buf_result_ptr, batch_size * num_heads * head_dim * sizeof(T));
  host_buf_result_ptr += batch_size * num_heads * head_dim * sizeof(T);
  memcpy(S_allreduce_h.data(), host_buf_result_ptr, batch_size * num_heads * sizeof(float));

  size_t V_num_results_error_atol_1e_3_rtol_1e_3 = 0, S_num_results_error_atol_1e_3_rtol_1e_3 = 0;
  bool V_nan_detected = false, S_nan_detected = false;

  for (uint32_t i = 0; i < batch_size * num_heads * head_dim; ++i) {
    if (std::isnan(float(V_allreduce_h[i]))) {
      V_nan_detected = true;
    }
    if (!utils::isclose(float(V_allreduce_h[i]), float(V_reduce_ref_h[i]), 1e-3, 1e-3)) {
      spdlog::error("rank: {}, i: {}, V_allreduce_h[i]: {}, V_reduce_ref_h[i]: {}", rank, i,
                    float(V_allreduce_h[i]), float(V_reduce_ref_h[i]));
      V_num_results_error_atol_1e_3_rtol_1e_3++;
    }
  }

  float V_accuracy = 1. - float(V_num_results_error_atol_1e_3_rtol_1e_3) /
                              float(batch_size * num_heads * head_dim);

  for (uint32_t i = 0; i < batch_size * num_heads; ++i) {
    if (std::isnan(float(S_allreduce_h[i]))) {
      S_nan_detected = true;
    }
    if (!utils::isclose(float(S_allreduce_h[i]), float(S_reduce_ref_h[i]), 1e-3, 1e-3)) {
      spdlog::error("rank: {}, i: {}, S_allreduce_h[i]: {}, S_reduce_ref_h[i]: {}, S[i]: {}", rank,
                    i, S_allreduce_h[i], S_reduce_ref_h[i], S[i]);
      S_num_results_error_atol_1e_3_rtol_1e_3++;
    }
  }

  float S_accuracy =
      1. - float(S_num_results_error_atol_1e_3_rtol_1e_3) / float(batch_size * num_heads);

  spdlog::info("rank: {}, V_accuracy: {}, S_accuracy: {}, V_nan_detected: {}, S_nan_detected: {}",
               rank, V_accuracy, S_accuracy, V_nan_detected, S_nan_detected);

  if (V_accuracy < 0.9 || S_accuracy < 0.99 || V_nan_detected || S_nan_detected) {
    spdlog::error("rank: {}, attention all reduce kernel failed.", rank);
  }

  MPI_Finalize();
  spdlog::info("MPI Finalized.");
  return 0;
}
