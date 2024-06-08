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
  const uint32_t batch_size = 128;
  const uint32_t num_heads = 64;
  const uint32_t head_dim = 128;
  const uint32_t aligned_buf_size_in_bytes =
      ceil_div(
          (batch_size * num_heads * head_dim) * sizeof(T) + batch_size * num_heads * sizeof(float),
          128) *
      128;
  thrust::device_vector<uint8_t> device_buff(aligned_buf_size_in_bytes);

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
                             thrust::raw_pointer_cast(device_buff.data()),
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
      thrust::raw_pointer_cast(device_buff.data()), rank, nranks, batch_size, num_heads, head_dim);

  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    spdlog::error("rank: {}, cuda error: {}", rank, cudaGetErrorString(status));
  }

  MPI_Finalize();
  spdlog::info("MPI Finalized.");
  return 0;
}