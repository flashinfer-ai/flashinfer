/*
 * Copyright (c) 2025 by SageAttention team.
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

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"

namespace nvfp4_attention {

using namespace cute;

template <int Depth>
using PipelineState = cutlass::PipelineState<Depth>;

template <int Stages>
class ProducerConsumerPipeline {
 public:
  using Pipeline = cutlass::PipelineTmaAsync<Stages>;
  using PipelineState = cutlass::PipelineState<Stages>;

 private:
  Pipeline pipeline_;

 public:
  template <typename SharedStorage>
  __device__ __forceinline__ ProducerConsumerPipeline(SharedStorage& shared_storage)
      : pipeline_(shared_storage.pipeline) {}

  __device__ __forceinline__ void producer_acquire(PipelineState& state) {
    pipeline_.producer_acquire(state);
  }

  __device__ __forceinline__ auto* producer_get_barrier(PipelineState& state) {
    return pipeline_.producer_get_barrier(state);
  }

  __device__ __forceinline__ void producer_commit(PipelineState& state) {
    pipeline_.producer_commit(state);
  }

  __device__ __forceinline__ void producer_tail(PipelineState& state) {
    pipeline_.producer_tail(state);
  }

  __device__ __forceinline__ auto consumer_try_wait(PipelineState& state) {
    return pipeline_.consumer_try_wait(state);
  }

  template <typename BarrierToken>
  __device__ __forceinline__ void consumer_wait(PipelineState& state,
                                                BarrierToken const& barrier_token) {
    pipeline_.consumer_wait(state, barrier_token);
  }

  __device__ __forceinline__ void consumer_wait(PipelineState& state) {
    auto token = consumer_try_wait(state);
    consumer_wait(state, token);
  }

  __device__ __forceinline__ void consumer_release(PipelineState& state) {
    pipeline_.consumer_release(state);
  }
};

template <int NumPipelines, int Stages>
class MultiPipelineManager {
 public:
  using Pipeline = ProducerConsumerPipeline<Stages>;
  using PipelineState = cutlass::PipelineState<Stages>;

 private:
  Pipeline* pipelines_[NumPipelines];

 public:
  __device__ __forceinline__ MultiPipelineManager(Pipeline* pipelines[NumPipelines]) {
    for (int i = 0; i < NumPipelines; ++i) {
      pipelines_[i] = pipelines[i];
    }
  }

  __device__ __forceinline__ Pipeline& get_pipeline(int idx) { return *pipelines_[idx]; }

  __device__ __forceinline__ void producer_tail_all(PipelineState states[NumPipelines]) {
    for (int i = 0; i < NumPipelines; ++i) {
      pipelines_[i]->producer_tail(states[i]);
    }
  }
};

template <int Stages>
__device__ __forceinline__ auto make_pipeline_state(int index = 0, bool phase = false,
                                                    int count = 0) {
  return PipelineState<Stages>{index, phase, count};
}

template <typename Pipeline, typename State, typename LoadFunc>
__device__ __forceinline__ void pipeline_producer_load(Pipeline& pipeline, State& state,
                                                       LoadFunc const& load_func) {
  pipeline.producer_acquire(state);

  load_func(pipeline.producer_get_barrier(state));

  pipeline.producer_commit(state);
  ++state;
}

template <typename Pipeline, typename State, typename ComputeFunc>
__device__ __forceinline__ void pipeline_consumer_compute(Pipeline& pipeline, State& state,
                                                          ComputeFunc const& compute_func) {
  pipeline.consumer_wait(state);

  compute_func();

  pipeline.consumer_release(state);
  ++state;
}

}  // namespace nvfp4_attention
