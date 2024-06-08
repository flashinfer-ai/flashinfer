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
#include <mpi.h>
#include <spdlog/spdlog.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/host_vector.h>

#include <cstdint>
#include <flashinfer/distributed/all_reduce.cuh>

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
  using T = float;
  using ReduceT = float;
  const size_t num_elems = 2 * 1024 * 1024;
  std::vector<T> host_buf(num_elems);
  for (uint32_t i = 0; i < num_elems; ++i) {
    host_buf[i] = T(i + rank);
  }
  thrust::device_vector<T> device_buf(host_buf);
  const size_t buf_size_in_bytes = num_elems * sizeof(T);

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
                             thrust::raw_pointer_cast(device_buf.data()), buf_size_in_bytes);
  std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> sm_channel_handlers(sm_channels.size());
  std::transform(
      sm_channels.begin(), sm_channels.end(), sm_channel_handlers.begin(),
      [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
  thrust::device_vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> sm_channel_handlers_d(
      sm_channel_handlers);
  spdlog::info("rank: {}, sm channels setup.", rank);

  MPI_Barrier(MPI_COMM_WORLD);

  // call attention all reduce kernel
  dim3 nthrs(1024);
  dim3 nblks(128);
  distributed::SumAllReduceInplaceKernel<T, ReduceT><<<nblks, nthrs>>>(
      thrust::raw_pointer_cast(sm_channel_handlers_d.data()),
      thrust::raw_pointer_cast(device_buf.data()), rank, nranks, device_buf.size());

  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    spdlog::error("rank: {}, cuda error: {}", rank, cudaGetErrorString(status));
  }

  // check result correctness
  thrust::host_vector<T> host_buf_result = device_buf;
  size_t num_results_error_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;

  for (uint32_t i = 0; i < num_elems; ++i) {
    T expected = T(i * nranks + (nranks - 1) * nranks / 2);
    if (std::isnan(float(host_buf_result[i]))) {
      nan_detected = true;
    }
    if (!utils::isclose(float(host_buf_result[i]), float(expected), 1e-3, 1e-3)) {
      num_results_error_atol_1e_3_rtol_1e_3++;
    }
  }
  float result_accuracy = 1. - float(num_results_error_atol_1e_3_rtol_1e_3) / float(num_elems);

  spdlog::info("rank: {}, nan_detected: {} accuracy: {}", rank, nan_detected, result_accuracy);
  if (result_accuracy < 0.99 || nan_detected) {
    spdlog::error("rank: {}, accuracy test failed.", rank);
  }

  MPI_Finalize();
  spdlog::info("MPI Finalized.");
  return 0;
}
