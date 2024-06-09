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
#ifndef FLASHINFER_DISTRIBUTED_ALL_REDUCE_CUH_
#define FLASHINFER_DISTRIBUTED_ALL_REDUCE_CUH_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/proxy_channel_device.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>

#include "../attention/state.cuh"
#include "../vec_dtypes.cuh"

namespace flashinfer {

namespace distributed {

void SetupChannels(mscclpp::Communicator* comm, std::vector<mscclpp::SmChannel>& sm_channels,
                   int rank, int nranks, void* buff, size_t buff_size_in_bytes) {
  const mscclpp::TransportFlags all_transports = mscclpp::Transport::CudaIpc;
  mscclpp::RegisteredMemory buf_reg_mem =
      comm->registerMemory(buff, buff_size_in_bytes, all_transports);

  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remote_reg_mem;
  std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> conn_futures;

  for (int r = 0; r < nranks; ++r) {
    if (r == rank) continue;

    mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
    conn_futures.push_back(comm->connectOnSetup(r, 0, transport));

    comm->sendMemoryOnSetup(buf_reg_mem, r, 0);
    auto remoteMemory = comm->recvMemoryOnSetup(r, 0);
    remote_reg_mem.push_back(remoteMemory);
  }
  comm->setup();
  std::transform(
      conn_futures.begin(), conn_futures.end(), std::back_inserter(connections),
      [](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>& future) {
        return future.get();
      });

  std::unordered_map<size_t, std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> sm_semaphores;
  for (size_t cid = 0; cid < connections.size(); ++cid) {
    sm_semaphores.emplace(
        cid, std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*comm, connections[cid]));
  }
  comm->setup();

  for (size_t cid = 0; cid < connections.size(); ++cid) {
    if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
      sm_channels.emplace_back(sm_semaphores[cid], remote_reg_mem[cid].get(), buf_reg_mem.data());
    }
  }
}

constexpr uint32_t MAX_RANKS = 8;
__device__ mscclpp::DeviceSyncer device_syncer;

template <typename DType>
__global__ void AttentionAllReduceInplaceKernel(mscclpp::SmChannelDeviceHandle* sm_channels,
                                                uint8_t* buf, const uint32_t rank,
                                                const uint32_t num_ranks, const uint32_t batch_size,
                                                const uint32_t num_heads, const uint32_t head_dim) {
  const uint32_t vec_size = 16 / sizeof(DType);
  const size_t chunk_size = head_dim / num_ranks;
  if (num_ranks == 1) return;
  const uint32_t num_peers = num_ranks - 1;
  const uint32_t tid = threadIdx.x + blockDim.x * (threadIdx.y + blockIdx.x * blockDim.y);
  const uint32_t tx = threadIdx.x;
  const uint32_t head_id = threadIdx.y;
  const uint32_t batch_id = blockIdx.x;
  DType* v_buf = (DType*)buf;
  float* s_buf = (float*)(buf + batch_size * num_heads * head_dim * sizeof(DType));

  if (tid < num_peers) {
    sm_channels[tid].signal();
    sm_channels[tid].wait();
  }
  device_syncer.sync(gridDim.x);

  float other_lse[MAX_RANKS - 1], self_lse = s_buf[batch_id * num_heads + head_id];
  for (uint32_t round_idx = 0; round_idx < num_peers; ++round_idx) {
    int peer_idx = (round_idx + rank);
    if (peer_idx >= num_peers) peer_idx -= num_peers;
    other_lse[round_idx] =
        ((float*)(sm_channels[peer_idx].dst_ + batch_size * num_heads * head_dim *
                                                   sizeof(DType)))[batch_id * num_heads + head_id];
  }

  device_syncer.sync(gridDim.x);
  if (tid < num_peers) {
    sm_channels[tid].signal();
    sm_channels[tid].wait();
  }
  device_syncer.sync(gridDim.x);

  state_t<vec_size> tmp;
  for (uint32_t elem_idx = tx; elem_idx < chunk_size / vec_size; elem_idx += blockDim.x) {
    tmp.init();
    tmp.o.cast_load(v_buf + (batch_id * num_heads + head_id) * head_dim + rank * chunk_size +
                    elem_idx * vec_size);
    tmp.m = self_lse;
    for (uint32_t round_idx = 0; round_idx < num_peers; ++round_idx) {
      int peer_idx = (round_idx + rank);
      if (peer_idx >= num_peers) peer_idx -= num_peers;
      vec_t<float, vec_size> other_v;
      other_v.cast_load(((DType*)sm_channels[peer_idx].dst_) +
                        (batch_id * num_heads + head_id) * head_dim + rank * chunk_size +
                        elem_idx * vec_size);
      tmp.merge(other_v, other_lse[round_idx], 1);
    }
    tmp.normalize();

    device_syncer.sync(gridDim.x);
    if (tid < num_peers) {
      sm_channels[tid].signal();
      sm_channels[tid].wait();
    }
    device_syncer.sync(gridDim.x);

    for (uint32_t round_idx = 0; round_idx < num_peers; ++round_idx) {
      int peer_idx = (round_idx + rank);
      if (peer_idx >= num_peers) peer_idx -= num_peers;
      tmp.o.cast_store(((DType*)sm_channels[peer_idx].dst_) +
                       (batch_id * num_heads + head_id) * head_dim + rank * chunk_size +
                       elem_idx * vec_size);
    }
    tmp.o.cast_store(v_buf + (batch_id * num_heads + head_id) * head_dim + rank * chunk_size +
                     elem_idx * vec_size);

    device_syncer.sync(gridDim.x);
    if (tid < num_peers) {
      sm_channels[tid].signal();
      sm_channels[tid].wait();
    }
    device_syncer.sync(gridDim.x);
  }
  float lse = tmp.get_lse();
  device_syncer.sync(gridDim.x);
  if (tid < num_peers) {
    sm_channels[tid].signal();
    sm_channels[tid].wait();
  }
  device_syncer.sync(gridDim.x);

  if (tx == 0) {
    for (uint32_t round_idx = 0; round_idx < num_peers; ++round_idx) {
      int peer_idx = (round_idx + rank);
      if (peer_idx >= num_peers) peer_idx -= num_peers;
      ((float*)(sm_channels[peer_idx].dst_ + batch_size * num_heads * head_dim *
                                                 sizeof(DType)))[batch_id * num_heads + head_id] =
          lse;
    }
    s_buf[batch_id * num_heads + head_id] = lse;
  }

  device_syncer.sync(gridDim.x);
  if (tid < num_peers) {
    sm_channels[tid].signal();
    sm_channels[tid].wait();
  }
}

template <typename DType, typename ReduceDType>
__global__ void SumAllReduceInplaceKernel(mscclpp::SmChannelDeviceHandle* sm_channels, DType* buf,
                                          const uint32_t rank, const uint32_t num_ranks,
                                          const size_t num_elems) {
  const uint32_t vec_size = 16 / sizeof(DType);
  const size_t chunk_size = num_elems / num_ranks;
  if (num_ranks == 1) return;
  const uint32_t num_peers = num_ranks - 1;
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < num_peers) {
    sm_channels[tid].signal();
    sm_channels[tid].wait();
  }
  device_syncer.sync(gridDim.x);

  size_t num_vec_per_chunk = chunk_size / vec_size;
  // use int4 as much as possible
  for (uint32_t i = tid; i < num_vec_per_chunk; i += blockDim.x * gridDim.x) {
    vec_t<ReduceDType, vec_size> tmp;
    tmp.cast_load(buf + rank * chunk_size + i * vec_size);
    for (uint32_t round_idx = 0; round_idx < num_peers; ++round_idx) {
      int peer_idx = (round_idx + rank);
      if (peer_idx >= num_peers) peer_idx -= num_peers;
      vec_t<ReduceDType, vec_size> val;
      val.cast_load(((DType*)sm_channels[peer_idx].dst_) + rank * chunk_size + i * vec_size);
#pragma unroll
      for (int j = 0; j < vec_size; ++j) {
        tmp[j] += val[j];
      }
    }

    device_syncer.sync(gridDim.x);
    if (tid < num_peers) {
      sm_channels[tid].signal();
      sm_channels[tid].wait();
    }
    device_syncer.sync(gridDim.x);

    for (uint32_t round_idx = 0; round_idx < num_peers; ++round_idx) {
      int peer_idx = (round_idx + rank);
      if (peer_idx >= num_peers) peer_idx -= num_peers;
      tmp.cast_store(((DType*)sm_channels[peer_idx].dst_) + rank * chunk_size + i * vec_size);
    }
    tmp.cast_store(buf + rank * chunk_size + i * vec_size);

    device_syncer.sync(gridDim.x);
    if (tid < num_peers) {
      sm_channels[tid].signal();
      sm_channels[tid].wait();
    }
    device_syncer.sync(gridDim.x);
  }

  device_syncer.sync(gridDim.x);
  if (tid < num_peers) {
    sm_channels[tid].signal();
    sm_channels[tid].wait();
  }
}

}  // namespace distributed

}  // namespace flashinfer

#endif  // FLASHINFER_DISTRIBUTED_ALL_REDUCE_CUH_
